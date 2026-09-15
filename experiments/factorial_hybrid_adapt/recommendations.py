"""
Étage 1 — Recommandations BUY / NEUTRAL / SELL (univers complet).

## Définition du score (existant, non modifié)
Les modèles C1–C4 prédisent ``alpha_ajuste_risque`` :
    alpha = (forward_ret_titre - forward_ret_MASI) / vol_baissiere_20d
Donc ``prediction`` est déjà un excès de rendement ajusté au risque vs MASI,
pas un rendement brut.

## Benchmark
Mode par défaut ``zero_alpha`` :
    benchmark_return = 0
    excess_return = prediction - 0 = prediction
Justification : sur l'échelle alpha, "battre le marché" ⇔ alpha > 0.

Mode alternatif ``cs_median`` (même jour, cross-sectionnel, pas de futur) :
    benchmark_return = médiane(prediction) le jour t
    excess_return = prediction - benchmark

## Règle (paramètre ``tau``)
    BUY     si excess_return >  +tau
    NEUTRAL si |excess_return| <= tau
    SELL    si excess_return <  -tau

``tau`` est calibré UNIQUEMENT sur la fenêtre train du bloc (ex. quantile
des |prediction| train), jamais sur le test — anti look-ahead.

## Conviction
    conviction_score = excess_return   (intensité continue, signée)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.metrics import evaluate_scoring_model

logger = logging.getLogger(__name__)

Recommendation = Literal["BUY", "NEUTRAL", "SELL"]
BenchmarkMode = Literal["zero_alpha", "cs_median"]
TauMethod = Literal["fixed", "train_abs_quantile", "expanding_past_quantile"]

CELL_META = {
    1: {"model": "ridge", "regime": False, "cell_name": "ridge_static", "label": "C1 Ridge"},
    2: {"model": "ridge", "regime": True, "cell_name": "ridge_adapt", "label": "C2 Ridge + régime"},
    3: {"model": "hybrid", "regime": False, "cell_name": "hybrid_static", "label": "C3 Hybrid"},
    4: {"model": "hybrid", "regime": True, "cell_name": "hybrid_adapt", "label": "C4 Hybrid + régime"},
}


@dataclass(frozen=True)
class RecommendationConfig:
    """
    Paramètres de la règle BUY/NEUTRAL/SELL (aucun look-ahead).

    ``tau`` est le seul seuil de conviction. Modes :
    - ``fixed`` : valeur a priori (grille de sensibilité possible hors test).
    - ``train_abs_quantile`` : quantile des |pred| sur le train du bloc
      (nécessite ``train_predictions_for_tau``).
    - ``expanding_past_quantile`` : quantile des |pred| des plis strictement
      antérieurs (même cellule) — causal quand seules les preds test WF
      sont disponibles.
    """

    benchmark_mode: BenchmarkMode = "zero_alpha"
    tau_method: TauMethod = "expanding_past_quantile"
    # Si tau_method=fixed : valeur absolue sur l'échelle alpha
    tau_fixed: float = 0.10
    # Quantile des |pred| (train ou passé expanding)
    tau_quantile: float = 0.33
    # Bornes de sécurité
    tau_min: float = 1e-4
    tau_max: float = 5.0


def calibrate_tau(
    reference_predictions: pd.Series,
    cfg: RecommendationConfig,
) -> float:
    """
    Calibre tau sur des prédictions de référence causales uniquement
    (train du bloc, ou historique des plis antérieurs — jamais le test courant).
    """
    if cfg.tau_method == "fixed":
        tau = float(cfg.tau_fixed)
    elif cfg.tau_method in ("train_abs_quantile", "expanding_past_quantile"):
        s = pd.to_numeric(reference_predictions, errors="coerce").dropna().abs()
        if s.empty:
            tau = float(cfg.tau_fixed)
        else:
            tau = float(s.quantile(cfg.tau_quantile))
    else:
        raise ValueError(f"tau_method inconnu : {cfg.tau_method}")
    return float(np.clip(tau, cfg.tau_min, cfg.tau_max))


def _benchmark_series(
    predictions: pd.Series,
    dates: pd.Series,
    mode: BenchmarkMode,
) -> pd.Series:
    if mode == "zero_alpha":
        return pd.Series(0.0, index=predictions.index)
    if mode == "cs_median":
        panel = pd.DataFrame({"date": pd.to_datetime(dates), "p": predictions})
        med = panel.groupby("date")["p"].transform("median")
        return med
    raise ValueError(f"benchmark_mode inconnu : {mode}")


def assign_recommendation(excess: float, tau: float) -> Recommendation:
    if excess > tau:
        return "BUY"
    if excess < -tau:
        return "SELL"
    return "NEUTRAL"


def month_end_mask(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    mois = d.dt.to_period("M")
    last = d.groupby(mois).transform("max")
    return d == last


def generate_recommendations(
    predictions: pd.DataFrame,
    *,
    train_predictions_for_tau: pd.Series | None = None,
    cfg: RecommendationConfig | None = None,
    rebalance_only: bool = True,
    pred_col: str = "prediction",
    date_col: str = "date_cours",
    ticker_col: str = "ticker",
    target_col: str = TARGET_COLUMN,
    tau_override: float | None = None,
) -> pd.DataFrame:
    """
    Transforme les scores continus en recommandations titre × date.

    Paramètres
    ----------
    predictions :
        Panel test (ou as-of) avec au minimum ticker, date, prediction.
        Colonnes optionnelles : cell_id, model, regime, fold_id, alpha_ajuste_risque.
    train_predictions_for_tau :
        Série de référence causale pour calibrer tau
        (train du bloc, ou historique expanding des plis antérieurs).
        Requis si tau_method in {train_abs_quantile, expanding_past_quantile}
        sauf si ``tau_override`` est fourni.
    cfg :
        Configuration benchmark / tau.
    rebalance_only :
        Si True, ne garde que les fins de mois (dates de rebalancement).
    tau_override :
        Si fourni, ignore la calibration (utile pour grille de sensibilité).
    """
    cfg = cfg or RecommendationConfig()
    df = predictions.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna(subset=[date_col, pred_col, ticker_col])

    if rebalance_only:
        df = df.loc[month_end_mask(df[date_col])].copy()

    if tau_override is not None:
        tau = float(np.clip(tau_override, cfg.tau_min, cfg.tau_max))
    elif cfg.tau_method == "fixed":
        tau = calibrate_tau(pd.Series(dtype=float), cfg)
    else:
        if train_predictions_for_tau is None or len(train_predictions_for_tau) == 0:
            # Fallback documenté : tau fixe a priori (jamais le test courant)
            logger.warning(
                "Référence tau vide (méthode=%s) — fallback tau_fixed=%.4f",
                cfg.tau_method,
                cfg.tau_fixed,
            )
            tau = float(cfg.tau_fixed)
        else:
            tau = calibrate_tau(train_predictions_for_tau, cfg)

    bench = _benchmark_series(df[pred_col], df[date_col], cfg.benchmark_mode)
    excess = df[pred_col].to_numpy(dtype=float) - bench.to_numpy(dtype=float)
    recos = np.where(
        excess > tau,
        "BUY",
        np.where(excess < -tau, "SELL", "NEUTRAL"),
    )

    out = pd.DataFrame(
        {
            "date": df[date_col].values,
            "ticker": df[ticker_col].astype(str).values,
            "predicted_return": df[pred_col].values,
            "benchmark_return": bench.values,
            "excess_return": excess,
            "conviction_score": excess,
            "tau": tau,
            "recommendation": recos,
        }
    )

    if "cell_id" in df.columns:
        out["cell"] = df["cell_id"].values
    for col in (
        "cell_name",
        "model",
        "regime",
        "fold_id",
        "test_start_month",
        "test_end_month",
    ):
        if col in df.columns:
            out[col] = df[col].values

    # Cible réalisée (évaluation uniquement — ne sert PAS à former la reco)
    if target_col in df.columns:
        out["realized_alpha"] = pd.to_numeric(df[target_col], errors="coerce").values

    out["benchmark_mode"] = cfg.benchmark_mode
    out["tau_method"] = cfg.tau_method
    return out.reset_index(drop=True)


def generate_recommendations_for_fold(
    fold_preds_test: pd.DataFrame,
    fold_preds_train: pd.DataFrame,
    *,
    cfg: RecommendationConfig | None = None,
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """Convenience : tau calibré sur train du pli, reco sur test (rebal mensuel)."""
    return generate_recommendations(
        fold_preds_test,
        train_predictions_for_tau=fold_preds_train[pred_col],
        cfg=cfg,
        rebalance_only=True,
        pred_col=pred_col,
    )


def build_recommendations_walk_forward(
    preds: pd.DataFrame,
    *,
    fold_ids: list[int] | None = None,
    cfg: RecommendationConfig | None = None,
    pred_col: str = "prediction",
    date_col: str = "date_cours",
) -> pd.DataFrame:
    """
    Construit le panel de recommandations pour un ensemble de plis WF.

    Pour ``expanding_past_quantile`` : tau du pli F = quantile des |pred|
    mois-fin des plis de la même cellule avec test_end < test_start(F).
    L'historique tau utilise **tous** les plis disponibles dans ``preds``
    (ex. 81), même si l'évaluation ne retient que ``fold_ids`` (ex. 14).
    """
    cfg = cfg or RecommendationConfig()
    all_preds = preds.copy()
    all_preds[date_col] = pd.to_datetime(all_preds[date_col], errors="coerce")
    all_preds[pred_col] = pd.to_numeric(all_preds[pred_col], errors="coerce")
    all_preds["test_end"] = pd.to_datetime(all_preds["test_end"], errors="coerce")
    all_preds["test_start"] = pd.to_datetime(all_preds["test_start"], errors="coerce")

    df = all_preds if fold_ids is None else all_preds[all_preds["fold_id"].isin(fold_ids)].copy()

    # Index causal des plis évalués (ordre chronologique par test_start)
    fold_meta = (
        df.groupby("fold_id", as_index=False)
        .agg(
            test_start=("test_start", "first"),
            test_end=("test_end", "first"),
            test_start_month=("test_start_month", "first"),
            test_end_month=("test_end_month", "first"),
        )
        .sort_values("test_start")
    )

    # Historique mois-fin pour tau : tous les plis (causal via test_end < t_start)
    me = all_preds.loc[
        month_end_mask(all_preds[date_col]),
        [date_col, pred_col, "cell_id", "fold_id", "test_end"],
    ].copy()

    frames: list[pd.DataFrame] = []
    for _, frow in fold_meta.iterrows():
        fid = int(frow["fold_id"])
        t_start = pd.Timestamp(frow["test_start"])
        fold_df = df[df["fold_id"] == fid]
        for cell_id, cell_df in fold_df.groupby("cell_id"):
            ref: pd.Series | None = None
            if cfg.tau_method == "expanding_past_quantile":
                past = me.loc[
                    (me["cell_id"] == int(cell_id)) & (me["test_end"] < t_start),
                    pred_col,
                ]
                ref = past
            elif cfg.tau_method == "train_abs_quantile":
                raise ValueError(
                    "build_recommendations_walk_forward ne fournit pas les preds train ; "
                    "utiliser generate_recommendations_for_fold ou tau_method="
                    "expanding_past_quantile|fixed."
                )
            frames.append(
                generate_recommendations(
                    cell_df,
                    train_predictions_for_tau=ref,
                    cfg=cfg,
                    rebalance_only=True,
                    pred_col=pred_col,
                    date_col=date_col,
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Évaluation étage 1
# ---------------------------------------------------------------------------


def _signal_metrics(df: pd.DataFrame, pred_col: str = "predicted_return") -> dict[str, float]:
    if "realized_alpha" not in df.columns or df.empty:
        return {
            "IC_mean": np.nan,
            "IC_median": np.nan,
            "IC_std": np.nan,
            "hit_ratio": np.nan,
            "pct_IC_pos": np.nan,
        }
    eval_df = df.rename(
        columns={"date": "date_cours", "realized_alpha": TARGET_COLUMN}
    ).copy()
    if pred_col != "prediction":
        eval_df["prediction"] = eval_df[pred_col]
    m = evaluate_scoring_model(
        eval_df,
        "prediction",
        TARGET_COLUMN,
        date_col="date_cours",
    )
    ics: list[float] = []
    for _, g in eval_df.groupby("date_cours"):
        sub = g[["prediction", TARGET_COLUMN]].dropna()
        if len(sub) < 5 or sub["prediction"].std() == 0 or sub[TARGET_COLUMN].std() == 0:
            continue
        ics.append(float(sub["prediction"].corr(sub[TARGET_COLUMN], method="spearman")))
    return {
        "IC_mean": m.get("rank_ic_mean", np.nan),
        "IC_median": float(np.median(ics)) if ics else np.nan,
        "IC_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan,
        "hit_ratio": m.get("hit_ratio", np.nan),
        "pct_IC_pos": float(np.mean([x > 0 for x in ics])) if ics else np.nan,
        "n_rebalance_dates": int(eval_df["date_cours"].nunique()),
        "n_rows": int(len(eval_df)),
    }


def _reco_quality(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {}
    vc = df["recommendation"].value_counts()
    out: dict[str, Any] = {
        "n_BUY": int(vc.get("BUY", 0)),
        "n_NEUTRAL": int(vc.get("NEUTRAL", 0)),
        "n_SELL": int(vc.get("SELL", 0)),
        "pct_BUY": float(vc.get("BUY", 0) / n),
        "pct_NEUTRAL": float(vc.get("NEUTRAL", 0) / n),
        "pct_SELL": float(vc.get("SELL", 0) / n),
    }

    # Précision : BUY correct si realized_alpha > 0 ; SELL correct si realized_alpha < 0
    if "realized_alpha" in df.columns:
        buy = df[df["recommendation"] == "BUY"]
        sell = df[df["recommendation"] == "SELL"]
        out["precision_BUY"] = (
            float((buy["realized_alpha"] > 0).mean()) if len(buy) else np.nan
        )
        out["precision_SELL"] = (
            float((sell["realized_alpha"] < 0).mean()) if len(sell) else np.nan
        )
        out["mean_realized_alpha_BUY"] = (
            float(buy["realized_alpha"].mean()) if len(buy) else np.nan
        )
        out["mean_realized_alpha_SELL"] = (
            float(sell["realized_alpha"].mean()) if len(sell) else np.nan
        )
        out["mean_realized_alpha_NEUTRAL"] = (
            float(df.loc[df["recommendation"] == "NEUTRAL", "realized_alpha"].mean())
            if (df["recommendation"] == "NEUTRAL").any()
            else np.nan
        )
    return out


def evaluate_stage1_recommendations(recos: pd.DataFrame) -> dict[str, Any]:
    """Agrège métriques signal + qualité des classes pour un panel de recommandations."""
    signal = _signal_metrics(recos)
    quality = _reco_quality(recos)
    return {**signal, **quality}


def look_ahead_checklist(cfg: RecommendationConfig) -> list[dict[str, str]]:
    """Contrôles documentés anti look-ahead (à afficher dans les rapports)."""
    return [
        {
            "check": "prediction_uses_only_features_at_t",
            "status": "OK",
            "detail": "Scores issus du WF : features *_z / régime connus à date_cours.",
        },
        {
            "check": "tau_not_fit_on_test",
            "status": "OK",
            "detail": (
                f"tau_method={cfg.tau_method} : calibré sur train du bloc, "
                "historique des plis antérieurs, ou fixe a priori — jamais sur le test courant."
            ),
        },
        {
            "check": "benchmark_available_at_t",
            "status": "OK",
            "detail": (
                f"benchmark_mode={cfg.benchmark_mode} : "
                "0 (échelle alpha) ou médiane CS du jour t — pas de rendement futur."
            ),
        },
        {
            "check": "recommendation_ignores_realized_alpha",
            "status": "OK",
            "detail": "realized_alpha n'entre pas dans assign_recommendation ; usage éval seulement.",
        },
        {
            "check": "score_definition",
            "status": "INFO",
            "detail": (
                "predicted_return = prédiction de alpha_ajuste_risque "
                "(excès vs MASI / vol baissière), pas un return brut. "
                "benchmark_return=0 sous zero_alpha."
            ),
        },
    ]


def compare_cells_table(recos: pd.DataFrame) -> pd.DataFrame:
    """Table synthétique C1–C4 à partir d'un panel de recommandations."""
    rows = []
    for cell_id in sorted(recos["cell"].dropna().unique()):
        sub = recos[recos["cell"] == cell_id]
        m = evaluate_stage1_recommendations(sub)
        meta = CELL_META.get(int(cell_id), {})
        rows.append(
            {
                "Configuration": meta.get("label", f"C{int(cell_id)}"),
                "cell": int(cell_id),
                "IC": m.get("IC_mean"),
                "Hit": m.get("hit_ratio"),
                "pct_IC_pos": m.get("pct_IC_pos"),
                "% BUY": m.get("pct_BUY"),
                "% Neutral": m.get("pct_NEUTRAL"),
                "% SELL": m.get("pct_SELL"),
                "Precision BUY": m.get("precision_BUY"),
                "Precision SELL": m.get("precision_SELL"),
                "mean_alpha_BUY": m.get("mean_realized_alpha_BUY"),
                "mean_alpha_SELL": m.get("mean_realized_alpha_SELL"),
                "tau_mean": float(sub["tau"].mean()) if "tau" in sub.columns else np.nan,
                "n_rows": m.get("n_rows"),
            }
        )
    return pd.DataFrame(rows)


def merge_fold_signal_metrics(
    compare_df: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    fold_ids: list[int],
) -> pd.DataFrame:
    """
    Remplace IC/Hit du panel mois-fin par les IC journaliers WF des plis retenus
    (plus fidèles à l'analyse principale déjà validée).
    """
    out = compare_df.copy()
    fm = fold_metrics[fold_metrics["fold_id"].isin(fold_ids)].copy()
    agg = (
        fm.groupby("cell_id")
        .agg(
            IC=("rank_ic", "mean"),
            Hit=("hit_ratio", "mean"),
            IC_median=("rank_ic", "median"),
            IC_std=("rank_ic", "std"),
            pct_IC_pos=("rank_ic", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .rename(columns={"cell_id": "cell"})
    )
    keep_reco = [c for c in out.columns if c not in ("IC", "Hit", "pct_IC_pos")]
    merged = out[keep_reco].merge(agg, on="cell", how="left")
    cols = [
        "Configuration",
        "cell",
        "IC",
        "IC_median",
        "IC_std",
        "Hit",
        "pct_IC_pos",
        "% BUY",
        "% Neutral",
        "% SELL",
        "Precision BUY",
        "Precision SELL",
        "mean_alpha_BUY",
        "mean_alpha_SELL",
        "tau_mean",
        "n_rows",
    ]
    return merged[[c for c in cols if c in merged.columns]]
