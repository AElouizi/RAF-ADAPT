"""Écriture features vers Supabase ou fichiers locaux."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from postgrest.exceptions import APIError
from supabase import Client

from bvc_recommender.config import (
    FEATURES_DIR,
    INDICATORS_MIN_DATE,
    TABLE_FEATURES_FONDAMENTALES,
    TABLE_FEATURES_INDICES,
    TABLE_FEATURES_TECHNIQUES,
)
from bvc_recommender.data.loader import fetch_table
from bvc_recommender.features.fundamental import sanitize_valuation_ratios

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.replace({float("nan"): None})
    return out.to_dict(orient="records")


def save_features_local(
    df: pd.DataFrame,
    name: str,
    output_dir: Path | None = None,
) -> Path:
    """Sauvegarde parquet (fallback csv)."""
    output_dir = output_dir or FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    logger.info("Sauvegardé localement : %s (%s lignes)", path, len(df))
    return path


def upsert_features(
    client: Client,
    table: str,
    df: pd.DataFrame,
    *,
    on_conflict: str = "ticker,date_fin",
) -> dict[str, Any]:
    """
    Tente un upsert Supabase par lots.

    Returns statut {ok, rows_written, error, fallback_local}.
    """
    result: dict[str, Any] = {
        "table": table,
        "ok": False,
        "rows_written": 0,
        "error": None,
        "fallback_local": None,
    }
    if df.empty:
        result["error"] = "dataframe vide"
        return result

    payload = _serialize_records(df)
    written = 0
    try:
        for i in range(0, len(payload), BATCH_SIZE):
            batch = payload[i : i + BATCH_SIZE]
            client.table(table).upsert(batch, on_conflict=on_conflict).execute()
            written += len(batch)
        result["ok"] = True
        result["rows_written"] = written
        logger.info("Upsert Supabase %s : %s lignes", table, written)
    except APIError as exc:
        result["error"] = str(exc)
        logger.warning("Upsert Supabase %s échoué : %s", table, exc)
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Upsert Supabase %s échoué : %s", table, exc)

    if not result["ok"]:
        local_name = table.replace("features_", "features_")
        result["fallback_local"] = str(
            save_features_local(df, local_name)
        )
    return result


def _purge_fundamental_tickers_not_in(client: Client, tickers: list[str]) -> int:
    """Supprime les lignes des tickers absents du run courant (résidus d'anciens calculs)."""
    if not tickers:
        return 0
    try:
        resp = client.table(TABLE_FEATURES_FONDAMENTALES).select("ticker").execute()
        rows = resp.data or []
        if not rows:
            return 0
        existing = {str(r["ticker"]) for r in rows if r.get("ticker")}
        allowed = {str(t) for t in tickers}
        stale = sorted(existing - allowed)
        removed = 0
        for ticker in stale:
            client.table(TABLE_FEATURES_FONDAMENTALES).delete().eq("ticker", ticker).execute()
            removed += 1
        if removed:
            logger.info(
                "Purge features_fondamentales : %s tickers obsolètes supprimés (%s)",
                removed,
                ", ".join(stale[:10]) + ("…" if len(stale) > 10 else ""),
            )
        return removed
    except Exception as exc:
        logger.warning("Purge features_fondamentales ignorée : %s", exc)
        return 0


def _nullify_invalid_valuation_ratios(client: Client) -> int:
    """Met à NULL les PE/PB ≤ 0 déjà présents en base (résidus d'anciens runs)."""
    try:
        df = fetch_table(
            client,
            TABLE_FEATURES_FONDAMENTALES,
            columns="ticker,date_fin,pe,pb",
            order_by="ticker",
        )
        if df.empty:
            return 0
        for col in ("pe", "pb"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid = df[
            (df["pe"].notna() & (df["pe"] <= 0))
            | (df["pb"].notna() & (df["pb"] <= 0))
        ].copy()
        if invalid.empty:
            return 0
        payload = []
        for _, row in invalid.iterrows():
            record: dict[str, Any] = {
                "ticker": row["ticker"],
                "date_fin": row["date_fin"],
            }
            if pd.notna(row.get("pe")) and row["pe"] <= 0:
                record["pe"] = None
            if pd.notna(row.get("pb")) and row["pb"] <= 0:
                record["pb"] = None
            payload.append(record)
        written = 0
        for i in range(0, len(payload), BATCH_SIZE):
            batch = payload[i : i + BATCH_SIZE]
            client.table(TABLE_FEATURES_FONDAMENTALES).upsert(
                batch,
                on_conflict="ticker,date_fin",
            ).execute()
            written += len(batch)
        logger.info(
            "Nettoyage features_fondamentales : %s lignes PE/PB invalides → NULL",
            written,
        )
        return written
    except Exception as exc:
        logger.warning("Nettoyage PE/PB Supabase ignoré : %s", exc)
        return 0


def _purge_before_min_date(client: Client, table: str, date_col: str) -> None:
    """Supprime les lignes antérieures au périmètre global (2011–2014, etc.)."""
    floor = INDICATORS_MIN_DATE
    try:
        before = (
            client.table(table)
            .select("ticker", count="exact")
            .lt(date_col, floor)
            .execute()
        )
        stale_count = before.count or 0
        if stale_count == 0:
            return
        client.table(table).delete().lt(date_col, floor).execute()
        after = (
            client.table(table)
            .select("ticker", count="exact")
            .lt(date_col, floor)
            .execute()
        )
        if (after.count or 0) > 0:
            logger.warning(
                "Purge %s avant %s incomplète (%s lignes restantes) — exécuter fix_rls_features.sql",
                table,
                floor,
                after.count,
            )
        else:
            logger.info("Purge %s : %s lignes supprimées (avant %s)", table, stale_count, floor)
    except Exception as exc:
        logger.warning("Purge %s avant %s ignorée : %s", table, floor, exc)


def _purge_pre_listing_rows(
    client: Client,
    table: str,
    date_col: str,
    listing_dates: pd.DataFrame,
    tickers: list[str],
) -> int:
    """Supprime en base les lignes antérieures à la 1ère cotation (résidus d'anciens runs)."""
    if not tickers or listing_dates.empty:
        return 0

    ld = listing_dates[listing_dates["ticker"].isin(tickers)].copy()
    ld["listing_date"] = pd.to_datetime(ld["listing_date"], errors="coerce")
    ld = ld.dropna(subset=["ticker", "listing_date"])
    purged = 0
    for _, row in ld.iterrows():
        ticker = str(row["ticker"])
        listing = pd.Timestamp(row["listing_date"]).strftime("%Y-%m-%d")
        try:
            before = (
                client.table(table)
                .select("ticker", count="exact")
                .eq("ticker", ticker)
                .lt(date_col, listing)
                .execute()
            )
            stale_count = before.count or 0
            if stale_count == 0:
                continue
            client.table(table).delete().eq("ticker", ticker).lt(date_col, listing).execute()
            after = (
                client.table(table)
                .select("ticker", count="exact")
                .eq("ticker", ticker)
                .lt(date_col, listing)
                .execute()
            )
            if (after.count or 0) > 0:
                logger.warning(
                    "Purge %s pré-cotation incomplète pour %s (%s lignes restantes) — "
                    "exécuter fix_rls_features.sql (policy DELETE)",
                    table,
                    ticker,
                    after.count,
                )
            else:
                purged += 1
                logger.info(
                    "Purge %s : %s lignes supprimées pour %s (avant %s)",
                    table,
                    stale_count,
                    ticker,
                    listing,
                )
        except Exception as exc:
            logger.warning(
                "Purge %s pré-cotation ignorée pour %s : %s",
                table,
                ticker,
                exc,
            )
    if purged:
        logger.info("Purge %s pré-cotation : %s tickers nettoyés", table, purged)
    return purged


def publish_features(
    client: Client | None,
    fundamental: pd.DataFrame,
    technical: pd.DataFrame,
    *,
    full_universe: bool = False,
    listing_dates: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Publie les deux tables features (Supabase ou local)."""
    now = datetime.now(timezone.utc).isoformat()
    if not fundamental.empty:
        fundamental = sanitize_valuation_ratios(fundamental.copy())
        fundamental["computed_at"] = now
    if not technical.empty:
        technical = technical.copy()
        technical["computed_at"] = now

    status: dict[str, Any] = {"fundamental": None, "technical": None}

    if client is None:
        status["fundamental"] = {
            "ok": False,
            "error": "client Supabase indisponible",
            "fallback_local": str(save_features_local(fundamental, "features_fondamentales")),
        }
        status["technical"] = {
            "ok": False,
            "error": "client Supabase indisponible",
            "fallback_local": str(save_features_local(technical, "features_techniques")),
        }
        return status

    status["fundamental"] = upsert_features(
        client,
        TABLE_FEATURES_FONDAMENTALES,
        fundamental,
        on_conflict="ticker,date_fin",
    )
    if (
        full_universe
        and status["fundamental"].get("ok")
        and not fundamental.empty
        and "ticker" in fundamental.columns
    ):
        status["fundamental"]["stale_tickers_purged"] = _purge_fundamental_tickers_not_in(
            client,
            fundamental["ticker"].dropna().astype(str).unique().tolist(),
        )
    if status["fundamental"].get("ok"):
        status["fundamental"]["invalid_ratios_nullified"] = _nullify_invalid_valuation_ratios(client)
        _purge_before_min_date(client, TABLE_FEATURES_FONDAMENTALES, "date_fin")
        if listing_dates is not None and not listing_dates.empty and not fundamental.empty:
            status["fundamental"]["pre_listing_purged"] = _purge_pre_listing_rows(
                client,
                TABLE_FEATURES_FONDAMENTALES,
                "date_fin",
                listing_dates,
                fundamental["ticker"].dropna().astype(str).unique().tolist(),
            )
    status["technical"] = upsert_features(
        client,
        TABLE_FEATURES_TECHNIQUES,
        technical,
        on_conflict="ticker,date_cours",
    )
    if status["technical"].get("ok"):
        _purge_before_min_date(client, TABLE_FEATURES_TECHNIQUES, "date_cours")
        if listing_dates is not None and not listing_dates.empty:
            if not technical.empty:
                tech_tickers = technical["ticker"].dropna().astype(str).unique().tolist()
            elif not fundamental.empty:
                tech_tickers = fundamental["ticker"].dropna().astype(str).unique().tolist()
            else:
                tech_tickers = []
            if tech_tickers:
                status["technical"]["pre_listing_purged"] = _purge_pre_listing_rows(
                    client,
                    TABLE_FEATURES_TECHNIQUES,
                    "date_cours",
                    listing_dates,
                    tech_tickers,
                )
    return status


def publish_market_context(
    client: Client | None,
    market_context: pd.DataFrame,
) -> dict[str, Any]:
    """Publie features_indices (Supabase ou local)."""
    now = datetime.now(timezone.utc).isoformat()
    if not market_context.empty:
        market_context = market_context.copy()
        market_context["computed_at"] = now

    if client is None:
        return {
            "ok": False,
            "error": "client Supabase indisponible",
            "fallback_local": str(save_features_local(market_context, "features_indices")),
        }

    return upsert_features(
        client,
        TABLE_FEATURES_INDICES,
        market_context,
        on_conflict="date_cours",
    )
