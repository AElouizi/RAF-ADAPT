"""
Sous-module A — détection du régime de marché (règle à seuils, 3 états).

Écrit is_bull / is_neutral / is_bear / is_rebalancing_date dans
features_indices (source de vérité quotidienne, locale + Supabase).
Archive l'ancien encodage HMM dans features_indices_hmm_archive.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step4
    python -m bvc_recommender.scripts.run_step4 --start-year 2015 --end-year 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATASET_DIR,
    FEATURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_TRAIN_END,
    TABLE_FEATURES_INDICES,
    load_env_file,
)
from bvc_recommender.data.loader import get_supabase_client  # noqa: E402
from bvc_recommender.features.market_context import build_market_context  # noqa: E402
from bvc_recommender.features.writer import (  # noqa: E402
    publish_market_context,
    save_features_local,
    upsert_features,
)
from bvc_recommender.models.regime_detector import (  # noqa: E402
    DOCUMENTED_THRESHOLDS,
    LEGACY_NEUTRAL_COLUMN,
    REBALANCE_FLAG_COLUMN,
    REGIME_COLUMNS,
    REGIME_INPUTS,
    aggregate_monthly_context,
    dominant_regime,
    enrich_market_context_with_regime,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HMM_ARCHIVE_TABLE = "features_indices_hmm_archive"
HMM_ARCHIVE_LOCAL = "features_indices_hmm_archive"
HMM_LEGACY_COLS = ["is_bull", "is_sideways", "is_bear"]


def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        path = directory / f"{name}{ext}"
        if path.is_file():
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


def load_market_context(use_processed: bool) -> pd.DataFrame:
    ctx = _load_parquet("features_indices", FEATURES_DIR)
    if not ctx.empty:
        return ctx

    if not use_processed or not (DATA_PROCESSED_DIR / "market_data_cours_historique.parquet").is_file():
        raise FileNotFoundError(
            "features_indices.parquet manquant. Lancer run_step3 --use-processed --use-features."
        )

    cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
    indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)
    return build_market_context(cours, indices)


def archive_hmm_encoding(market_context: pd.DataFrame) -> Path | None:
    """
    Snapshot local (+ Supabase si possible) de l'ancien one-hot HMM
    avant remplacement par la règle à seuils.
    """
    df = market_context.copy()
    has_hmm = all(c in df.columns for c in HMM_LEGACY_COLS)
    if not has_hmm:
        # Déjà migré (is_neutral) sans sideways — rien à archiver depuis ce panel
        logger.info("Pas de colonnes HMM à archiver dans le panel courant.")
        return None

    # Ne pas ré-archiver si le panel est déjà au format seuils (is_neutral présent
    # et cohérent) — on archive uniquement si is_sideways existe sans is_neutral
    # OU si on n'a pas encore de fichier archive.
    archive_path = FEATURES_DIR / f"{HMM_ARCHIVE_LOCAL}.parquet"
    if archive_path.is_file():
        logger.info("Archive HMM locale déjà présente : %s (conservation)", archive_path)
        local_path = archive_path
    else:
        keep = ["date_cours"]
        for c in ("masi_mom_3m", "return_dispersion", "breadth_ma50", *HMM_LEGACY_COLS, REBALANCE_FLAG_COLUMN):
            if c in df.columns:
                keep.append(c)
        arch = df[keep].copy()
        arch["date_cours"] = pd.to_datetime(arch["date_cours"], errors="coerce")
        arch["archived_at"] = datetime.now(timezone.utc).isoformat()
        arch["note"] = "GaussianHMM 3-states snapshot before threshold regime"
        local_path = save_features_local(arch, HMM_ARCHIVE_LOCAL, FEATURES_DIR)
        logger.info("Archive HMM locale écrite : %s (%s j)", local_path, len(arch))

        try:
            client = get_supabase_client()
            pub = upsert_features(
                client,
                HMM_ARCHIVE_TABLE,
                arch.drop(columns=["archived_at", "note"], errors="ignore"),
                on_conflict="date_cours",
            )
            logger.info("Archive HMM Supabase : %s", pub)
        except Exception as exc:
            logger.warning("Archive HMM Supabase ignorée : %s", exc)

    return local_path


def save_regime_history(df: pd.DataFrame) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASET_DIR / "regime_history.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = DATASET_DIR / "regime_history.csv"
        df.to_csv(path, index=False)
    logger.info("Historique régime mensuel sauvegardé : %s (%s mois)", path, len(df))
    return path


def plot_regime_evolution(df: pd.DataFrame, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib indisponible — graphique non généré.")
        return None

    plot_df = df.dropna(subset=REGIME_COLUMNS).copy()
    if plot_df.empty:
        return None

    plot_df["x"] = range(len(plot_df))
    plot_df["regime_code"] = plot_df.apply(dominant_regime, axis=1)
    code_map = {"bull": 2, "neutral": 1, "sideways": 1, "bear": 0, "unknown": np.nan}
    plot_df["y"] = plot_df["regime_code"].map(code_map)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {"bull": "#2ecc71", "neutral": "#f39c12", "bear": "#e74c3c"}
    for regime, color in colors.items():
        mask = plot_df["regime_code"] == regime
        ax.scatter(
            plot_df.loc[mask, "x"],
            plot_df.loc[mask, "y"],
            c=color,
            label={"bull": "Haussier", "neutral": "Neutre", "bear": "Baissier"}[regime],
            s=28,
            zorder=3,
        )
    ax.plot(plot_df["x"], plot_df["y"], color="#94a3b8", linewidth=0.8, zorder=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Baissier", "Neutre", "Haussier"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("Évolution mensuelle du régime de marché BVC — règle à seuils")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    tick_pos = plot_df["x"].tolist()
    tick_labels = plot_df["month_label"].tolist()
    step = max(1, len(tick_labels) // 12)
    ax.set_xticks(tick_pos[::step])
    ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Graphique régime : %s", output_path)
    return output_path


def build_step4_report(
    regime_df: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    history_path: Path,
    chart_path: Path | None,
    features_path: Path | None,
    archive_path: Path | None,
) -> dict[str, Any]:
    valid = regime_df.dropna(subset=REGIME_COLUMNS)
    summary_rows = []
    for row in valid.itertuples(index=False):
        series = pd.Series(row._asdict())
        dom = dominant_regime(series)
        summary_rows.append(
            {
                "month": row.month_label,
                "as_of_date": str(pd.Timestamp(row.as_of_date).date()),
                "is_bull": int(getattr(row, "is_bull", 0) or 0),
                "is_neutral": int(getattr(row, "is_neutral", 0) or 0),
                "is_bear": int(getattr(row, "is_bear", 0) or 0),
                "dominant": dom,
            }
        )

    dom_counts = pd.Series([r["dominant"] for r in summary_rows]).value_counts().to_dict()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Sous-module A",
        "random_state": RANDOM_STATE,
        "method": (
            f"Règle à seuils both-AND (q={0.33}/{0.67}) — calibres train ≤ {SPLIT_TRAIN_END} "
            "(EOM mensuel), appliqués hors train ; remplace GaussianHMM"
        ),
        "thresholds_documented": DOCUMENTED_THRESHOLDS,
        "inputs": list(REGIME_INPUTS),
        "outputs": list(REGIME_COLUMNS) + [REBALANCE_FLAG_COLUMN, LEGACY_NEUTRAL_COLUMN],
        "source_of_truth": "features_indices (quotidien)",
        "hmm_archive": str(archive_path) if archive_path else None,
        "period": {"start_year": start_year, "end_year": end_year},
        "train_end": SPLIT_TRAIN_END,
        "months": len(valid),
        "dominant_regime_counts": dom_counts,
        "history_path": str(history_path),
        "features_indices_path": str(features_path) if features_path else None,
        "chart_path": str(chart_path) if chart_path else None,
        "timeline": summary_rows,
        "latest": summary_rows[-1] if summary_rows else None,
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    thr = report.get("thresholds_documented") or {}
    lines = [
        "# Évolution mensuelle du régime de marché (Sous-module A — seuils)",
        "",
        f"- **Généré** : {report['generated_at']}",
        f"- **Méthode** : {report['method']}",
        f"- **Période** : {report['period']['start_year']} → {report['period']['end_year']}",
        f"- **Mois** : {report['months']}",
        "",
        "## Seuils documentés (train 2015-2020)",
        "",
        f"- p33_momentum = {thr.get('p33_momentum')}",
        f"- p67_momentum = {thr.get('p67_momentum')}",
        f"- p33_breadth = {thr.get('p33_breadth')}",
        f"- p67_breadth = {thr.get('p67_breadth')}",
        "",
        "## Répartition des régimes",
        "",
    ]
    for regime, count in report.get("dominant_regime_counts", {}).items():
        lines.append(f"- **{regime}** : {count}")

    if report.get("latest"):
        last = report["latest"]
        label = {"bull": "Haussier", "neutral": "Neutre", "bear": "Baissier"}.get(
            last["dominant"], last["dominant"]
        )
        lines.extend(
            [
                "",
                "## Dernier mois",
                "",
                f"- **{last['month']}** (au {last['as_of_date']})",
                f"- **Régime** : {label}",
                f"- One-hot : bull={last['is_bull']} | neutral={last['is_neutral']} | bear={last['is_bear']}",
            ]
        )

    if report.get("chart_path"):
        lines.extend(["", f"![Régime]({Path(report['chart_path']).name})", ""])

    lines.extend(
        [
            "",
            "## Historique mensuel",
            "",
            "| Mois | Régime | is_bull | is_neutral | is_bear |",
            "|------|--------|---------|------------|---------|",
        ]
    )
    for row in report.get("timeline", []):
        lines.append(
            f"| {row['month']} | {row['dominant']} | {row['is_bull']} | "
            f"{row['is_neutral']} | {row['is_bear']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 4 (seuils)")
    parser.add_argument("--use-processed", action="store_true", help="Recalculer contexte si besoin")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--skip-supabase",
        action="store_true",
        help="Ne pas publier features_indices vers Supabase",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    logger.info(
        "Détection régime à seuils %s → %s (calibres ≤ %s) | thr=%s",
        args.start_year,
        args.end_year,
        SPLIT_TRAIN_END,
        DOCUMENTED_THRESHOLDS,
    )

    market_context = load_market_context(args.use_processed)
    market_context = market_context.drop(
        columns=[c for c in ("mu_bull", "mu_sideways", "mu_bear") if c in market_context.columns],
        errors="ignore",
    )

    archive_path = archive_hmm_encoding(market_context)

    enriched = enrich_market_context_with_regime(market_context)
    features_path = save_features_local(enriched, "features_indices", FEATURES_DIR)

    if not args.skip_supabase:
        try:
            client = get_supabase_client()
            # Upsert : is_neutral (+ miroir is_sideways). Si colonne absente en base,
            # retenter sans is_neutral (exécuter alter_features_indices_threshold_regime.sql).
            pub = publish_market_context(client, enriched)
            if not pub.get("ok"):
                err = str(pub.get("error") or "")
                if "is_neutral" in err:
                    slim = enriched.drop(columns=["is_neutral"], errors="ignore")
                    pub = publish_market_context(client, slim)
                    logger.warning(
                        "Colonne is_neutral absente en Supabase — upsert via is_sideways "
                        "(miroir). Exécuter scripts/sql/alter_features_indices_threshold_regime.sql"
                    )
            logger.info("Publication features_indices Supabase (%s) : %s", TABLE_FEATURES_INDICES, pub)
        except Exception as exc:
            logger.warning("Publication Supabase ignorée : %s", exc)

    regime_df = aggregate_monthly_context(
        enriched,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if regime_df.empty or regime_df.dropna(subset=REGIME_COLUMNS).empty:
        logger.error("Aucun régime calculé — vérifier features_indices.")
        return 2

    history_path = save_regime_history(regime_df)
    chart_path = plot_regime_evolution(regime_df, REPORTS_DIR / "regime_evolution.png")

    report = build_step4_report(
        regime_df,
        start_year=args.start_year,
        end_year=args.end_year,
        history_path=history_path,
        chart_path=chart_path,
        features_path=features_path,
        archive_path=archive_path,
    )
    json_path = REPORTS_DIR / "step4_validation_report.json"
    md_path = REPORTS_DIR / "regime_evolution.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    logger.info("Rapport étape 4 : %s", json_path)
    if report.get("latest"):
        last = report["latest"]
        logger.info(
            "Dernier mois %s : régime=%s (bull=%s neutral=%s bear=%s)",
            last["month"],
            last["dominant"],
            last["is_bull"],
            last["is_neutral"],
            last["is_bear"],
        )

    for regime, count in report.get("dominant_regime_counts", {}).items():
        logger.info("  %s : %s mois", regime, count)

    logger.info("Étape 4 terminée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
