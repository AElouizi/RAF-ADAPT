"""Vérification multi-projets MASI — rapport court pour l'utilisateur.

Scanne les .env du dossier parent « Modèle empirique », sonde
market_data_indices_historique sur chaque projet Supabase connu,
liste les tables PostgREST liées aux indices, et imprime du SQL
à coller dans le dashboard Supabase pour confirmation manuelle.

Usage:
    python -m bvc_recommender.scripts.verify_masi_user

Projet additionnel sans .env (ex. pungyebagtntbocoewir):
    set VERIFY_EXTRA_SUPABASE_URL=https://pungyebagtntbocoewir.supabase.co
    set VERIFY_EXTRA_SUPABASE_ANON_KEY=eyJ...
    python -m bvc_recommender.scripts.verify_masi_user
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from supabase import Client, ClientOptions, create_client  # noqa: E402

TABLE = "market_data_indices_historique"
ALT_TABLES = [
    "masi_historique",
    "market_data_masi",
    "indices_masi",
    "masi_historical",
    "masi_history",
    "indices",
    "index_historique",
    "market_indices",
    "masi",
    "historique_masi",
    "market_data_index",
]

KNOWN_EMPTY_REF = "pungyebagtntbocoewir"
CURRENT_REF = "tlrborylhfwxrrryyism"

SQL_DASHBOARD = """
-- Coller dans SQL Editor (Dashboard -> SQL -> New query)
-- Projet: {ref}
-- https://supabase.com/dashboard/project/{ref}/sql/new

SELECT COUNT(*) AS total_lignes
FROM public.market_data_indices_historique;

SELECT code_index, COUNT(*) AS n, MIN(date_index) AS dmin, MAX(date_index) AS dmax
FROM public.market_data_indices_historique
GROUP BY code_index
ORDER BY n DESC;

SELECT COUNT(*) AS masi_lignes, MIN(date_index) AS masi_min, MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

SELECT COUNT(*) AS masi_avant_2022
FROM public.market_data_indices_historique
WHERE code_index = 'MASI' AND date_index < '2022-01-01';

SELECT COUNT(*) AS toutes_lignes_avant_2020
FROM public.market_data_indices_historique
WHERE date_index < '2020-01-01';

SELECT date_index, valeur_index, scraped_at
FROM public.market_data_indices_historique
WHERE code_index = 'MASI'
ORDER BY date_index ASC
LIMIT 5;
"""


@dataclass
class SupabaseProject:
    ref: str
    url: str
    anon_key: str | None = None
    env_sources: list[str] = field(default_factory=list)


def _parse_env_file(path: Path) -> tuple[str | None, str | None]:
    url = key = None
    if not path.is_file():
        return url, key
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if "SUPABASE_URL" in k and "supabase.co" in v:
            url = v
        if k in (
            "NEXT_PUBLIC_SUPABASE_ANON_KEY",
            "SUPABASE_ANON_KEY",
            "SUPABASE_KEY",
        ) and v.startswith("eyJ"):
            key = v
    return url, key


def discover_projects() -> dict[str, SupabaseProject]:
    projects: dict[str, SupabaseProject] = {}
    env_files = sorted(PARENT.rglob(".env*"))
    for path in env_files:
        if path.suffix == ".example":
            continue
        if path.name not in {".env", ".env.txt"}:
            continue
        url, key = _parse_env_file(path)
        if not url:
            continue
        m = re.search(r"https://([^.]+)\.supabase\.co", url)
        ref = m.group(1) if m else url
        if ref not in projects:
            projects[ref] = SupabaseProject(ref=ref, url=url, anon_key=key)
        projects[ref].env_sources.append(str(path))
        if key:
            projects[ref].anon_key = key

    extra_url = os.getenv("VERIFY_EXTRA_SUPABASE_URL", "").strip()
    extra_key = os.getenv("VERIFY_EXTRA_SUPABASE_ANON_KEY", "").strip()
    if extra_url:
        m = re.search(r"https://([^.]+)\.supabase\.co", extra_url)
        ref = m.group(1) if m else extra_url
        if ref not in projects:
            projects[ref] = SupabaseProject(
                ref=ref,
                url=extra_url,
                anon_key=extra_key or None,
                env_sources=["VERIFY_EXTRA_SUPABASE_*"],
            )
        elif extra_key:
            projects[ref].anon_key = extra_key

    if KNOWN_EMPTY_REF not in projects:
        projects[KNOWN_EMPTY_REF] = SupabaseProject(
            ref=KNOWN_EMPTY_REF,
            url=f"https://{KNOWN_EMPTY_REF}.supabase.co",
            anon_key=None,
            env_sources=["(ancien .env — absent des fichiers actuels)"],
        )
    return projects


def _client(url: str, key: str) -> Client:
    verify = os.getenv("SUPABASE_VERIFY_SSL", "false").strip().lower() not in {
        "false",
        "0",
        "no",
    }
    return create_client(
        url,
        key,
        ClientOptions(httpx_client=httpx.Client(verify=verify)),
    )


def count_table(client: Client, table: str, **filters) -> int | None:
    q = client.table(table).select("*", count="exact").limit(0)
    for col, val in filters.items():
        if col.endswith("__gte"):
            q = q.gte(col[:-5], val)
        elif col.endswith("__lte"):
            q = q.lte(col[:-5], val)
        elif col.endswith("__lt"):
            q = q.lt(col[:-4], val)
        else:
            q = q.eq(col, val)
    try:
        return q.execute().count
    except Exception:
        return None


def min_max_masi(client: Client) -> tuple[str | None, str | None]:
    try:
        rmin = (
            client.table(TABLE)
            .select("date_index")
            .eq("code_index", "MASI")
            .order("date_index")
            .limit(1)
            .execute()
        )
        rmax = (
            client.table(TABLE)
            .select("date_index")
            .eq("code_index", "MASI")
            .order("date_index", desc=True)
            .limit(1)
            .execute()
        )
        dmin = rmin.data[0]["date_index"] if rmin.data else None
        dmax = rmax.data[0]["date_index"] if rmax.data else None
        return dmin, dmax
    except Exception:
        return None, None


def probe_indices_table(client: Client | None, project: SupabaseProject) -> dict:
    if not project.anon_key:
        return {
            "status": "SKIP",
            "reason": "Pas de clé anon — définir VERIFY_EXTRA_SUPABASE_ANON_KEY ou ajouter au .env",
        }
    if client is None:
        try:
            client = _client(project.url, project.anon_key)
        except Exception as exc:
            return {"status": "ERROR", "reason": str(exc)}

    total = count_table(client, TABLE)
    masi = count_table(client, TABLE, code_index="MASI")
    pre2020 = count_table(client, TABLE, date_index__lt="2020-01-01")
    masi_pre2022 = count_table(
        client, TABLE, code_index="MASI", date_index__lt="2022-01-01"
    )
    dmin, dmax = min_max_masi(client)
    covers_2010 = bool(dmin and dmin <= "2010-12-31")

    status = "OK"
    if total == 0:
        status = "EMPTY"
    elif masi == 0:
        status = "NO_MASI"

    return {
        "status": status,
        "total_rows": total,
        "masi_rows": masi,
        "masi_date_min": dmin,
        "masi_date_max": dmax,
        "rows_before_2020": pre2020,
        "masi_before_2022": masi_pre2022,
        "masi_covers_2010": covers_2010,
    }


def probe_alt_tables(client: Client) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ALT_TABLES:
        try:
            resp = client.table(name).select("*", count="exact").limit(1).execute()
            out[name] = {"exists": True, "count": resp.count}
        except Exception as exc:
            msg = str(exc)
            out[name] = {
                "exists": False,
                "error": msg[:120],
            }
    return out


def openapi_masi_tables(url: str, key: str) -> list[str]:
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/openapi+json",
    }
    verify = os.getenv("SUPABASE_VERIFY_SSL", "false").strip().lower() not in {
        "false",
        "0",
        "no",
    }
    try:
        r = httpx.get(
            url.rstrip("/") + "/rest/v1/",
            headers=headers,
            verify=verify,
            timeout=20,
        )
        if r.status_code != 200:
            return []
        paths = r.json().get("paths", {})
        tables = sorted(p.lstrip("/") for p in paths if p != "/" and not p.startswith("/rpc"))
        return [t for t in tables if "masi" in t.lower() or "indic" in t.lower()]
    except Exception:
        return []


def print_report(projects: dict[str, SupabaseProject], probes: dict[str, dict], alt: dict) -> None:
    print("=" * 72)
    print("RAPPORT MASI — vérification multi-projets")
    print("=" * 72)

    print("\n1) URLs Supabase distinctes (scan .env « Modèle empirique »)\n")
    for ref, proj in sorted(projects.items()):
        key_ok = "clé anon OK" if proj.anon_key else "PAS DE CLÉ"
        print(f"  - {ref}")
        print(f"    URL : {proj.url}")
        print(f"    {key_ok}")
        for src in proj.env_sources:
            print(f"    source : {src}")

    print("\n2) Probe REST `market_data_indices_historique`\n")
    print(f"  {'Projet':<28} {'Total':>8} {'MASI':>8} {'MASI min':>12} {'MASI max':>12} {'2010+':>6}")
    print("  " + "-" * 68)
    winner = None
    for ref, proj in sorted(projects.items()):
        p = probes[ref]
        if p.get("status") == "SKIP":
            print(f"  {ref:<28} {'—':>8} {'—':>8} {'—':>12} {'—':>12} {'?':>6}  (SKIP: {p.get('reason', '')[:40]})")
            continue
        if p.get("status") == "ERROR":
            print(f"  {ref:<28} {'ERR':>8} {'ERR':>8} {'—':>12} {'—':>12} {'?':>6}")
            continue
        total = p.get("total_rows")
        masi = p.get("masi_rows")
        dmin = p.get("masi_date_min") or "—"
        dmax = p.get("masi_date_max") or "—"
        ok = "OUI" if p.get("masi_covers_2010") else "NON"
        marker = " [actuel .env]" if ref == CURRENT_REF else ""
        print(
            f"  {ref:<28} {total or 0:>8} {masi or 0:>8} {dmin:>12} {dmax:>12} {ok:>6}{marker}"
        )
        if p.get("masi_covers_2010"):
            winner = ref

    print("\n3) Tables alternatives (projet actuel `.env` final)\n")
    if alt.get("openapi"):
        print(f"  OpenAPI (indices/masi) : {', '.join(alt['openapi']) or '(aucune)'}")
    for name, info in sorted(alt.get("probes", {}).items()):
        if info.get("exists"):
            print(f"  - {name} : EXISTE (count={info.get('count')})")
        else:
            print(f"  - {name} : absent")

    print("\n4) Verdict\n")
    if winner:
        print(f"  MASI 2010+ trouvé sur : {winner}")
        if winner != CURRENT_REF:
            print(f"  -> Mettre a jour `.env` (Projet_Cursor_RS_final) avec l'URL/cle de `{winner}`")
            print("  -> OU migrer les lignes MASI vers tlrborylhfwxrrryyism puis relancer run_step1")
        else:
            print("  -> Donnees OK sur le projet actuel ; relancer run_step1 si parquet obsolete")
    else:
        accessible = [
            ref
            for ref, p in probes.items()
            if p.get("status") not in ("SKIP", "ERROR") and projects[ref].anon_key
        ]
        if accessible:
            print("  Aucun projet accessible via .env n'a MASI depuis 2010.")
            print(f"  Projet actuel ({CURRENT_REF}) : MASI commence le 2022-11-21 (741 lignes).")
        if KNOWN_EMPTY_REF in projects and not projects[KNOWN_EMPTY_REF].anon_key:
            print(
                f"  `{KNOWN_EMPTY_REF}` : ancien .env vide (0 ligne historique) — "
                "clé absente des fichiers ; probe impossible sans VERIFY_EXTRA_SUPABASE_ANON_KEY."
            )
        print("\n  Si vous voyez MASI 2010+ dans le dashboard :")
        print("  - Verifier le project-ref en haut a gauche (tlrborylhfwxrrryyism ?)")
        print("  - Executer le SQL section 5 ci-dessous dans CE projet")
        print("  - Importer via : python -m bvc_recommender.scripts.import_masi_history")

    print("\n5) SQL à coller dans le dashboard (confirmation manuelle)\n")
    target = winner or CURRENT_REF
    print(SQL_DASHBOARD.format(ref=target).strip())

    print("\n6) .env recommandé (Projet_Cursor_RS_final)\n")
    current = projects.get(CURRENT_REF)
    if current and current.anon_key:
        print(f"  NEXT_PUBLIC_SUPABASE_URL=\"{current.url}\"")
        print("  NEXT_PUBLIC_SUPABASE_ANON_KEY=\"<clé anon du dashboard>\"")
        print("  SUPABASE_VERIFY_SSL=false")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Vérification MASI multi-projets Supabase")
    parser.add_argument(
        "--env",
        type=Path,
        default=ROOT / ".env",
        help="Fichier .env du projet final (défaut: Projet_Cursor_RS_final/.env)",
    )
    args = parser.parse_args()
    load_env_file(args.env)

    projects = discover_projects()
    probes: dict[str, dict] = {}
    alt: dict = {"probes": {}, "openapi": []}

    current = projects.get(CURRENT_REF)
    current_client = None
    if current and current.anon_key:
        current_client = _client(current.url, current.anon_key)
        alt["probes"] = probe_alt_tables(current_client)
        alt["openapi"] = openapi_masi_tables(current.url, current.anon_key)

    for ref, proj in sorted(projects.items()):
        client = current_client if ref == CURRENT_REF and current_client else None
        if client is None and proj.anon_key:
            try:
                client = _client(proj.url, proj.anon_key)
            except Exception:
                client = None
        probes[ref] = probe_indices_table(client, proj)

    print_report(projects, probes, alt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
