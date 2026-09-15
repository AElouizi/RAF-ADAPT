"""
Export des indicateurs vers Excel et Supabase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from supabase import Client

from bvc_recommender.config import (
    EXPORTS_DIR,
    FEATURES_DIR,
    INDICATORS_EXCEL_PATH,
)
from bvc_recommender.features.writer import publish_features, publish_market_context

logger = logging.getLogger(__name__)


def _load_feature(name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{name}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    csv = FEATURES_DIR / f"{name}.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return pd.DataFrame()


def load_all_indicators() -> dict[str, pd.DataFrame]:
    return {
        "features_fondamentales": _load_feature("features_fondamentales"),
        "features_techniques": _load_feature("features_techniques"),
        "features_indices": _load_feature("features_indices"),
    }


def export_indicators_excel(
    indicators: dict[str, pd.DataFrame] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Export multi-feuilles Excel des indicateurs calculés."""
    indicators = indicators or load_all_indicators()
    output_path = output_path or INDICATORS_EXCEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_map = {
        "features_fondamentales": "Fondamentales",
        "features_techniques": "Techniques",
        "features_indices": "Contexte_Marche",
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        meta = pd.DataFrame(
            [
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "features_fondamentales_rows": len(indicators.get("features_fondamentales", [])),
                    "features_techniques_rows": len(indicators.get("features_techniques", [])),
                    "features_indices_rows": len(indicators.get("features_indices", [])),
                }
            ]
        )
        meta.to_excel(writer, sheet_name="Meta", index=False)

        for key, sheet in sheet_map.items():
            df = indicators.get(key, pd.DataFrame())
            if df.empty:
                pd.DataFrame({"info": [f"Aucune donnée — lancer run_step2/run_step3"]}).to_excel(
                    writer, sheet_name=sheet, index=False
                )
            else:
                export_df = df.copy()
                for col in export_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
                    export_df[col] = export_df[col].dt.strftime("%Y-%m-%d")
                export_df.to_excel(writer, sheet_name=sheet, index=False)

    logger.info("Excel indicateurs : %s", output_path)
    return output_path


def publish_indicators_supabase(client: Client | None) -> dict[str, Any]:
    """Publie les 3 tables d'indicateurs vers Supabase."""
    indicators = load_all_indicators()
    status: dict[str, Any] = {
        "fundamental": None,
        "technical": None,
        "market_context": None,
    }

    pub = publish_features(
        client,
        indicators.get("features_fondamentales", pd.DataFrame()),
        indicators.get("features_techniques", pd.DataFrame()),
    )
    status["fundamental"] = pub.get("fundamental")
    status["technical"] = pub.get("technical")
    status["market_context"] = publish_market_context(
        client,
        indicators.get("features_indices", pd.DataFrame()),
    )
    return status


def export_and_publish(
    client: Client | None = None,
    *,
    excel: bool = True,
    supabase: bool = True,
    excel_path: Path | None = None,
) -> dict[str, Any]:
    """Export Excel + publication Supabase."""
    indicators = load_all_indicators()
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "excel_path": None,
        "supabase": None,
        "row_counts": {k: len(v) for k, v in indicators.items()},
    }

    if excel:
        result["excel_path"] = str(export_indicators_excel(indicators, excel_path))

    if supabase:
        result["supabase"] = publish_indicators_supabase(client)

    return result
