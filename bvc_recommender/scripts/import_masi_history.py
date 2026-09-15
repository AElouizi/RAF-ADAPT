"""
Import historique MASI (CSV / Parquet) vers Supabase.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.import_masi_history --file chemin/vers/masi.csv --dry-run
    python -m bvc_recommender.scripts.import_masi_history --file chemin/vers/masi.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from postgrest.exceptions import APIError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    TABLE_INDICES_HISTORIQUE,
    get_settings,
    load_env_file,
)
from bvc_recommender.data.loader import get_supabase_client  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TABLE = TABLE_INDICES_HISTORIQUE
ON_CONFLICT = "code_index,date_index"
BATCH_SIZE = 500

SCHEMA_COLUMNS = {
    "code_index",
    "nom_index",
    "date_index",
    "valeur_index",
    "valeur_haut",
    "valeur_bas",
    "variation_veille",
    "variation_annee",
    "capitalisation_marchande",
    "drupal_id",
    "scraped_at",
}

DATE_ALIASES = ("date_index", "date", "Date", "DATE")
VALUE_ALIASES = ("valeur_index", "close", "valeur", "value", "close_price", "Close")
CODE_ALIASES = ("code_index", "code", "Code", "CODE")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _parse_api_error(exc: APIError) -> dict[str, Any]:
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0]
    return {"message": str(exc)}


def _rls_hint(payload: dict[str, Any]) -> str | None:
    code = str(payload.get("code", ""))
    message = str(payload.get("message", "")).lower()
    if code in {"42501", "PGRST301"} or "row-level security" in message or "permission denied" in message:
        return (
            "Écriture refusée par RLS (Row Level Security). "
            "Dans Supabase SQL Editor, activez une policy INSERT/UPDATE pour le rôle anon "
            f"sur `{TABLE}`, par exemple :\n"
            f"  CREATE POLICY \"anon_upsert_{TABLE}\" ON public.{TABLE}\n"
            "    FOR ALL TO anon USING (true) WITH CHECK (true);\n"
            "Ou utilisez la clé service_role (hors dépôt) pour les imports batch."
        )
    return None


def load_input_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Format non supporté : {suffix} (attendu .csv ou .parquet)")


def normalize_indices(
    df: pd.DataFrame,
    *,
    default_code: str = "MASI",
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Fichier vide — aucune ligne à importer.")

    date_col = _pick_column(df, DATE_ALIASES)
    value_col = _pick_column(df, VALUE_ALIASES)
    if not date_col:
        raise ValueError(
            f"Colonne date introuvable. Attendu : {', '.join(DATE_ALIASES)}. "
            f"Colonnes présentes : {list(df.columns)}"
        )
    if not value_col:
        raise ValueError(
            f"Colonne valeur introuvable. Attendu : {', '.join(VALUE_ALIASES)}. "
            f"Colonnes présentes : {list(df.columns)}"
        )

    out = df.copy()
    code_col = _pick_column(out, CODE_ALIASES)
    if code_col:
        out["code_index"] = out[code_col].astype(str).str.strip().str.upper()
    else:
        out["code_index"] = default_code

    out["date_index"] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["valeur_index"] = pd.to_numeric(out[value_col], errors="coerce")

    extra = [
        c
        for c in out.columns
        if c in SCHEMA_COLUMNS and c not in {"code_index", "date_index", "valeur_index"}
    ]
    out = out[["code_index", "date_index", "valeur_index", *extra]]

    before = len(out)
    out = out.dropna(subset=["date_index", "valeur_index", "code_index"])
    dropped = before - len(out)
    if dropped:
        logger.warning("%s lignes ignorées (date ou valeur manquante).", dropped)

    dupes = out.duplicated(subset=["code_index", "date_index"], keep="last").sum()
    if dupes:
        logger.warning("%s doublons (code_index, date_index) — conservation de la dernière occurrence.", dupes)
    out = out.drop_duplicates(subset=["code_index", "date_index"], keep="last")
    out = out.sort_values(["code_index", "date_index"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Aucune ligne valide après normalisation.")
    return out


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    payload = df.copy()
    for col in payload.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        payload[col] = payload[col].dt.strftime("%Y-%m-%d")
    payload = payload.replace({float("nan"): None})
    return payload.to_dict(orient="records")


def upsert_indices(
    client,
    df: pd.DataFrame,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "table": TABLE,
        "dry_run": dry_run,
        "rows_total": len(df),
        "rows_written": 0,
        "ok": False,
        "error": None,
        "rls_hint": None,
    }

    if dry_run:
        result["ok"] = True
        result["preview"] = {
            "date_min": str(df["date_index"].min()),
            "date_max": str(df["date_index"].max()),
            "codes": sorted(df["code_index"].unique().tolist()),
            "sample_head": _serialize_records(df.head(5)),
            "sample_tail": _serialize_records(df.tail(3)),
        }
        return result

    records = _serialize_records(df)
    written = 0
    try:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            client.table(TABLE).upsert(batch, on_conflict=ON_CONFLICT).execute()
            written += len(batch)
            logger.info("Lot %s–%s / %s envoyé.", i + 1, i + len(batch), len(records))
        result["ok"] = True
        result["rows_written"] = written
    except APIError as exc:
        payload = _parse_api_error(exc)
        result["error"] = json.dumps(payload, ensure_ascii=False)
        result["rls_hint"] = _rls_hint(payload)
        logger.error("Upsert échoué : %s", payload.get("message", exc))
        if result["rls_hint"]:
            logger.error(result["rls_hint"])
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Upsert échoué : %s", exc)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Importe l'historique MASI (CSV/Parquet) vers market_data_indices_historique."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Chemin vers le fichier CSV ou Parquet.",
    )
    parser.add_argument(
        "--code-index",
        default="MASI",
        help="code_index par défaut si absent du fichier (défaut : MASI).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prévisualise la normalisation sans écrire dans Supabase.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Taille des lots upsert (défaut : {BATCH_SIZE}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = args.file.resolve()

    if not path.is_file():
        logger.error("Fichier introuvable : %s", path)
        return 1

    global BATCH_SIZE
    BATCH_SIZE = max(1, args.batch_size)

    try:
        raw = load_input_file(path)
        normalized = normalize_indices(raw, default_code=args.code_index.strip().upper())
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Normalisé : %s lignes | codes=%s | %s → %s",
        len(normalized),
        sorted(normalized["code_index"].unique().tolist()),
        normalized["date_index"].min(),
        normalized["date_index"].max(),
    )

    if args.dry_run:
        status = upsert_indices(None, normalized, dry_run=True)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        print("\n[DRY-RUN] Aucune écriture Supabase.")
        return 0

    try:
        get_settings().validate()
        client = get_supabase_client()
    except (ValueError, ModuleNotFoundError) as exc:
        logger.error("Configuration Supabase invalide : %s", exc)
        return 2

    status = upsert_indices(client, normalized, dry_run=False)
    print(json.dumps(status, indent=2, ensure_ascii=False))

    if status["ok"]:
        logger.info("Import terminé : %s lignes upsertées.", status["rows_written"])
        return 0

    if status.get("rls_hint"):
        print(f"\n{status['rls_hint']}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
