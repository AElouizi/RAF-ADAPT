"""Connexion Supabase et chargement paginé des 4 tables principales."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import certifi
import httpx
import pandas as pd
from postgrest.exceptions import APIError
from supabase import Client, ClientOptions, create_client

from bvc_recommender.config import (
    COURS_MIN_DATE,
    HISTO_CONST_IND_CSV,
    HISTO_CONST_IND_CSV_LEGACY,
    INDICATEURS_FINANCIERS_CSV,
    TABLE_COURS_HISTORIQUE,
    TABLE_DONNEES_FINANCIERS,
    TABLE_FONDAMENTAUX_RS,
    TABLE_FONDAMENTAUX_RS_BIS,
    TABLE_HISTO_CONST_IND,
    TABLE_INDICES_HISTORIQUE,
    TABLE_INDICATEURS_FINANCIERS,
    get_settings,
)
from bvc_recommender.data.masi_coverage import check_masi_coverage

logger = logging.getLogger(__name__)


def _configure_ca_env() -> None:
    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


def _build_httpx_client(verify_ssl: bool) -> httpx.Client:
    if verify_ssl:
        verify: ssl.SSLContext | bool = ssl.create_default_context(
            cafile=certifi.where()
        )
    else:
        warnings.warn(
            "SUPABASE_VERIFY_SSL=false : vérification TLS désactivée.",
            stacklevel=2,
        )
        verify = False
    return httpx.Client(
        verify=verify,
        timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0),
    )


def _create_supabase_client(api_key: str) -> Client:
    settings = get_settings()
    _configure_ca_env()
    options = ClientOptions(httpx_client=_build_httpx_client(settings.verify_ssl))
    return create_client(settings.supabase_url, api_key, options)


@lru_cache
def get_supabase_client() -> Client:
    """Client Supabase REST (clé anon / publishable)."""
    settings = get_settings()
    settings.validate()
    return _create_supabase_client(settings.supabase_key)


@lru_cache
def get_service_role_client() -> Client | None:
    """Client service_role (bypass RLS) si SUPABASE_SERVICE_ROLE_KEY est configurée."""
    settings = get_settings()
    key = settings.service_role_key
    if not key or not key.startswith("eyJ"):
        return None
    if not settings.supabase_url:
        return None
    return _create_supabase_client(key)


MASI_RPC_FUNCTION = "fetch_masi_historique"


def fetch_masi_via_rpc(client: Client, *, page_size: int = 1000) -> pd.DataFrame | None:
    """
    Charge le MASI complet via RPC SECURITY DEFINER (bypass RLS table).

    Nécessite enable_masi_rpc.sql exécuté une fois dans Supabase SQL Editor.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        try:
            end = offset + page_size - 1
            response = (
                client.rpc(MASI_RPC_FUNCTION)
                .range(offset, end)
                .execute()
            )
        except Exception as exc:
            if offset == 0:
                logger.debug("RPC %s indisponible : %s", MASI_RPC_FUNCTION, exc)
                return None
            break
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(0.05)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.info("RPC %s : %s lignes MASI chargées", MASI_RPC_FUNCTION, len(df))
    return df


def merge_masi_into_indices(indices: pd.DataFrame, masi_full: pd.DataFrame) -> pd.DataFrame:
    """Remplace les lignes MASI partielles par l'historique complet."""
    if indices.empty:
        return masi_full.copy()
    if masi_full.empty:
        return indices.copy()
    base = indices.loc[indices["code_index"].astype(str).str.upper() != "MASI"].copy()
    merged = pd.concat([base, masi_full], ignore_index=True)
    if "date_index" in merged.columns:
        merged["date_index"] = pd.to_datetime(merged["date_index"], errors="coerce")
        merged = merged.sort_values(["code_index", "date_index"]).reset_index(drop=True)
    return merged


def fetch_table_postgres(table: str) -> pd.DataFrame | None:
    """
    Charge une table via connexion Postgres directe (DATABASE_URL).

    Contourne les policies RLS — utile quand le SQL Editor voit plus de lignes
    que l'API REST anon (ex. MASI depuis 2010).
    """
    settings = get_settings()
    db_url = settings.database_url
    if not db_url or db_url.startswith("#"):
        return None
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "psycopg2 absent — pip install psycopg2-binary pour charger via DATABASE_URL"
        )
        return None
    try:
        conn = psycopg2.connect(db_url)
        try:
            df = pd.read_sql_query(f'SELECT * FROM public."{table}"', conn)
        finally:
            conn.close()
        logger.info("Postgres : %s lignes chargées depuis %s", len(df), table)
        return df
    except Exception as exc:
        logger.warning("Échec chargement Postgres %s : %s", table, exc)
        return None


def load_indices_historique(
    client: Client,
    *,
    max_rows: int | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """
    Charge market_data_indices_historique en priorisant la couverture MASI depuis 2010.

    Ordre de tentative si MASI incomplet via REST anon :
    1. Postgres (DATABASE_URL) — vérité SQL Editor
    2. REST service_role (bypass RLS)
    3. REST anon (résultat partiel, avec avertissement)
    """
    cache_path = cache_dir / "market_data_indices_historique.parquet" if cache_dir else None
    if cache_path and cache_path.is_file() and not refresh:
        df = pd.read_parquet(cache_path)
        masi = check_masi_coverage(df)
        if masi["covers_from_2010"]:
            logger.info("Cache indices OK — MASI depuis %s", masi["date_min"])
            return df
        logger.info(
            "Cache indices : MASI depuis %s seulement — rechargement pour couverture 2010+",
            masi["date_min"],
        )

    candidates: list[tuple[str, pd.DataFrame]] = []

    def _fetch_rest(rest_client: Client, label: str) -> None:
        logger.info("Chargement %s via REST (%s)...", TABLE_INDICES_HISTORIQUE, label)
        df_rest = fetch_table(
            rest_client,
            TABLE_INDICES_HISTORIQUE,
            max_rows=max_rows,
            checkpoint_path=checkpoint_path,
        )
        candidates.append((label, df_rest))

    _fetch_rest(client, "anon")

    masi_check = check_masi_coverage(candidates[-1][1])
    if not masi_check["covers_from_2010"]:
        df_pg = fetch_table_postgres(TABLE_INDICES_HISTORIQUE)
        if df_pg is not None and not df_pg.empty:
            candidates.append(("postgres", df_pg))

    if not any(check_masi_coverage(df)["covers_from_2010"] for _, df in candidates):
        sr_client = get_service_role_client()
        if sr_client is not None:
            _fetch_rest(sr_client, "service_role")

    best_label, best_df = candidates[0]
    for label, df in candidates:
        if check_masi_coverage(df)["covers_from_2010"]:
            best_label, best_df = label, df
            break
    else:
        best_label, best_df = max(
            candidates,
            key=lambda item: check_masi_coverage(item[1])["row_count"],
        )

    if not check_masi_coverage(best_df)["covers_from_2010"]:
        masi_rpc = fetch_masi_via_rpc(client)
        if masi_rpc is not None and not masi_rpc.empty:
            merged = merge_masi_into_indices(best_df, masi_rpc)
            if check_masi_coverage(merged)["covers_from_2010"]:
                best_label, best_df = "rpc_fetch_masi_historique", merged

    masi_final = check_masi_coverage(best_df)
    logger.info(
        "Indices chargés via %s — %s lignes, MASI %s → %s (%s lignes, couverture 2010=%s)",
        best_label,
        len(best_df),
        masi_final["date_min"],
        masi_final["date_max"],
        masi_final["row_count"],
        masi_final["covers_from_2010"],
    )
    if not masi_final["covers_from_2010"]:
        logger.warning(
            "MASI incomplet via API — %s. Ajoutez DATABASE_URL ou SUPABASE_SERVICE_ROLE_KEY "
            "dans .env, ou exécutez scripts/sql/fix_rls_indices.sql dans Supabase.",
            masi_final["notes"][0] if masi_final["notes"] else "",
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        best_df.to_parquet(cache_path, index=False)
        logger.info("Cache écrit %s", cache_path)

    return best_df


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[dict[str, Any]], int]:
    meta_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file() or not meta_path.is_file():
        return [], 0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(checkpoint_path)
    return df.to_dict(orient="records"), int(meta.get("offset", len(df)))


def _save_checkpoint(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    offset: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
    meta_path = checkpoint_path.with_suffix(".json")
    meta_path.write_text(json.dumps({"offset": offset}), encoding="utf-8")


def _clear_checkpoint(checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    for path in (checkpoint_path, checkpoint_path.with_suffix(".json")):
        if path.is_file():
            path.unlink()


def fetch_table(
    client: Client,
    table: str,
    *,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
    gte_filters: dict[str, Any] | None = None,
    order_by: str = "id",
    page_size: int = 1000,
    max_rows: int | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Charge une table Supabase par pagination (PostgREST range)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    if checkpoint_path is not None:
        rows, offset = _load_checkpoint(checkpoint_path)
        if rows:
            logger.info(
                "Reprise checkpoint %s — %s lignes (offset=%s)",
                table,
                len(rows),
                offset,
            )
    while True:
        limit = page_size
        if max_rows is not None:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            limit = min(limit, remaining)

        end = offset + limit - 1
        query = (
            client.table(table)
            .select(columns, count="exact")
            .order(order_by)
            .range(offset, end)
        )
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        if gte_filters:
            for key, value in gte_filters.items():
                query = query.gte(key, value)

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                response = query.execute()
                break
            except APIError as exc:
                payload = exc.args[0] if exc.args else {}
                if not isinstance(payload, dict):
                    payload = getattr(exc, "json", None) or getattr(exc, "details", None) or {}
                code = str(payload.get("code", ""))
                if code == "57014" and page_size > 50:
                    logger.warning(
                        "Timeout PostgREST sur %s — page_size %s → %s",
                        table,
                        page_size,
                        page_size // 2,
                    )
                    return fetch_table(
                        client,
                        table,
                        columns=columns,
                        filters=filters,
                        gte_filters=gte_filters,
                        order_by=order_by,
                        page_size=page_size // 2,
                        max_rows=max_rows,
                        checkpoint_path=checkpoint_path,
                    )
                if attempt >= max_attempts:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(
                    "APIError sur %s (offset=%s, code=%s) — retry dans %ss",
                    table,
                    offset,
                    code or payload.get("message", "?"),
                    wait,
                )
                time.sleep(wait)
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
            ):
                if checkpoint_path is not None and rows:
                    _save_checkpoint(checkpoint_path, rows, offset)
                if attempt >= max_attempts:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(
                    "Erreur réseau sur %s (offset=%s, tentative %s/%s) — retry dans %ss",
                    table,
                    offset,
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(f"Échec chargement {table} après {max_attempts} tentatives")

        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, rows, offset)
        if len(batch) < limit or (max_rows is not None and len(rows) >= max_rows):
            break
        time.sleep(0.05)

    _clear_checkpoint(checkpoint_path)
    return pd.DataFrame(rows)


def _resolve_table_name(client: Client, candidates: list[str]) -> str:
    """Retourne le premier nom de table accessible parmi les candidats."""
    for name in candidates:
        try:
            client.table(name).select("*", count="exact").limit(1).execute()
            return name
        except Exception:
            continue
    return candidates[0]


def load_all_tables(
    client: Client | None = None,
    *,
    max_financial_rows: int | None = None,
    max_cours_rows: int | None = None,
    max_indices_rows: int | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Charge les 4 tables Supabase de l'étape 1.

    Tables :
    - fondamentaux_rs / fondamentaux_RS
    - donnees_financieres (alias brief : donnees_financiers)
    - market_data_cours_historique
    - market_data_indices_historique
    """
    client = client or get_supabase_client()

    fond_table = _resolve_table_name(
        client, [TABLE_FONDAMENTAUX_RS, "fondamentaux_RS", "fondamentaux_rs"]
    )
    fin_table = _resolve_table_name(
        client, [TABLE_DONNEES_FINANCIERS, "donnees_financiers", "donnees_financieres"]
    )

    def _load_or_cache(key: str, log_msg: str, fetch_kwargs: dict[str, Any]) -> pd.DataFrame:
        cache_path = cache_dir / f"{key}.parquet" if cache_dir else None
        if cache_path and cache_path.is_file() and not refresh:
            logger.info("Cache hit %s (%s)", key, cache_path)
            return pd.read_parquet(cache_path)
        logger.info(log_msg)
        if checkpoint_dir is not None:
            fetch_kwargs = {
                **fetch_kwargs,
                "checkpoint_path": checkpoint_dir / f"{key}.parquet",
            }
        df = fetch_table(client, **fetch_kwargs)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Cache écrit %s (%s lignes)", key, len(df))
        return df

    checkpoint_dir = cache_dir / "_checkpoints" if cache_dir else None

    fondamentaux = _load_or_cache(
        "fondamentaux_rs",
        f"Chargement {fond_table}...",
        {"table": fond_table},
    )

    donnees_financieres = _load_or_cache(
        "donnees_financieres",
        f"Chargement {fin_table} (statut_validation=VALIDE)...",
        {
            "table": fin_table,
            "filters": {"statut_validation": "VALIDE"},
            "page_size": 200,
            "max_rows": max_financial_rows,
        },
    )

    cours = _load_or_cache(
        "market_data_cours_historique",
        f"Chargement {TABLE_COURS_HISTORIQUE} (>= {COURS_MIN_DATE})...",
        {
            "table": TABLE_COURS_HISTORIQUE,
            "gte_filters": {"date_cours": COURS_MIN_DATE},
            "max_rows": max_cours_rows,
        },
    )

    checkpoint_indices = checkpoint_dir / "market_data_indices_historique.parquet" if checkpoint_dir else None
    indices = load_indices_historique(
        client,
        max_rows=max_indices_rows,
        cache_dir=cache_dir,
        refresh=refresh,
        checkpoint_path=checkpoint_indices,
    )

    return {
        "fondamentaux_rs": fondamentaux,
        "donnees_financieres": donnees_financieres,
        "market_data_cours_historique": cours,
        "market_data_indices_historique": indices,
    }


def probe_connection(client: Client | None = None) -> dict[str, dict[str, Any]]:
    """Teste la lecture d'une ligne par table (diagnostic connexion)."""
    client = client or get_supabase_client()
    tables = [
        TABLE_FONDAMENTAUX_RS,
        TABLE_DONNEES_FINANCIERS,
        TABLE_COURS_HISTORIQUE,
        TABLE_INDICES_HISTORIQUE,
    ]
    results: dict[str, dict[str, Any]] = {}
    for table in tables:
        try:
            response = (
                client.table(table)
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
            results[table] = {
                "ok": True,
                "count": response.count,
                "columns": list(response.data[0].keys()) if response.data else [],
            }
        except Exception as exc:
            results[table] = {"ok": False, "error": str(exc)}
    return results


def _load_csv_fallback(path: Path, *, sep: str = ";") -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)


def _load_histo_csv_fallback() -> pd.DataFrame:
    """Charge histo_const_ind depuis export CSV local (priorité au fichier avec colonne ticker)."""
    for path, sep in (
        (HISTO_CONST_IND_CSV, ";"),
        (HISTO_CONST_IND_CSV_LEGACY, ";"),
    ):
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(path, sep=sep)
        if not df.empty and "ticker" in df.columns:
            logger.info("histo_const_ind depuis CSV local (%s, %s lignes)", path.name, len(df))
            return df
        if not df.empty:
            logger.warning(
                "CSV %s chargé (%s lignes) sans colonne ticker — ignoré",
                path.name,
                len(df),
            )
    return pd.DataFrame()


def load_histo_const_ind(
    client: Client | None = None,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Charge histo_const_ind (cours, nombre_titre) — Supabase ou CSV local."""
    cache_path = cache_dir / "histo_const_ind.parquet" if cache_dir else None
    if cache_path and cache_path.is_file() and not refresh:
        logger.info("Cache hit histo_const_ind (%s)", cache_path)
        return pd.read_parquet(cache_path)

    df = pd.DataFrame()
    if client is not None:
        try:
            df = fetch_table(client, TABLE_HISTO_CONST_IND)
        except Exception as exc:
            logger.warning("histo_const_ind Supabase : %s", exc)

    if df.empty:
        df = _load_histo_csv_fallback()

    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def load_fondamentaux_rs_bis(
    client: Client | None = None,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Charge la table large fondamentaux_rs_bis (source de référence des ratios).

    Cette table n'a pas de colonne ``id`` : on ordonne par ``ticker``.
    """
    cache_path = cache_dir / "fondamentaux_rs_bis.parquet" if cache_dir else None
    if cache_path and cache_path.is_file() and not refresh:
        logger.info("Cache hit fondamentaux_rs_bis (%s)", cache_path)
        return pd.read_parquet(cache_path)

    df = pd.DataFrame()
    if client is not None:
        try:
            df = fetch_table(client, TABLE_FONDAMENTAUX_RS_BIS, order_by="ticker")
        except Exception as exc:
            logger.warning("fondamentaux_rs_bis Supabase : %s", exc)

    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def load_indicateurs_financiers(
    client: Client | None = None,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Charge le référentiel indicateurs_financiers — Supabase ou CSV local."""
    cache_path = cache_dir / "indicateurs_financiers.parquet" if cache_dir else None
    if cache_path and cache_path.is_file() and not refresh:
        logger.info("Cache hit indicateurs_financiers (%s)", cache_path)
        return pd.read_parquet(cache_path)

    df = pd.DataFrame()
    if client is not None:
        try:
            df = fetch_table(client, TABLE_INDICATEURS_FINANCIERS)
        except Exception as exc:
            logger.warning("indicateurs_financiers Supabase : %s", exc)

    if df.empty:
        df = _load_csv_fallback(INDICATEURS_FINANCIERS_CSV)
        if not df.empty:
            logger.info("indicateurs_financiers depuis CSV local (%s lignes)", len(df))

    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df
