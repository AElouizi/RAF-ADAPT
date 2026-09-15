"""
Bloc B — métriques niveau 1 / 2, format long, test omnibus, bootstrap stationnaire.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.metrics import (
    _cross_sectional_ic,
    evaluate_scoring_model,
)
from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_GROUP
from experiments.factorial_hybrid_adapt.recommendations import (
    RecommendationConfig,
    generate_recommendations,
    month_end_mask,
)

logger = logging.getLogger(__name__)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    err = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(err**2)))


def level1_predictive_metrics(
    pred_df: pd.DataFrame,
    *,
    pred_col: str = "prediction",
    target_col: str = TARGET_COLUMN,
    date_col: str = "date_cours",
) -> dict[str, float]:
    """Rank-IC, IC-IR, RMSE sur le panel test du pli."""
    m = evaluate_scoring_model(pred_df, pred_col, target_col, date_col=date_col)
    rank_ics = _cross_sectional_ic(
        pred_df, pred_col, target_col, date_col, method="spearman"
    )
    rank_ic = float(np.mean(rank_ics)) if rank_ics else float("nan")
    ic_std = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else float("nan")
    ic_ir = (
        float(rank_ic / ic_std)
        if np.isfinite(rank_ic) and np.isfinite(ic_std) and ic_std > 0
        else float("nan")
    )
    rmse = _rmse(
        pred_df[target_col].to_numpy(dtype=float),
        pred_df[pred_col].to_numpy(dtype=float),
    )
    return {
        "rank_ic": rank_ic,
        "ic_ir": ic_ir,
        "rmse": rmse,
        "hit_ratio_sign": float(m.get("hit_ratio", np.nan)),
    }


def _reco_turnover(recos: pd.DataFrame) -> float:
    """
    Turnover des recommandations : part moyenne des titres dont la classe
    change d'une fin de mois à la suivante (hors première date).
    """
    if recos.empty or "recommendation" not in recos.columns:
        return float("nan")
    df = recos.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return 0.0
    rates: list[float] = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = df.loc[df["date"] == d0, ["ticker", "recommendation"]].drop_duplicates(
            "ticker"
        )
        b = df.loc[df["date"] == d1, ["ticker", "recommendation"]].drop_duplicates(
            "ticker"
        )
        merged = a.merge(b, on="ticker", suffixes=("_0", "_1"))
        if merged.empty:
            continue
        rates.append(float((merged["recommendation_0"] != merged["recommendation_1"]).mean()))
    return float(np.mean(rates)) if rates else float("nan")


def level2_selection_metrics(
    pred_df: pd.DataFrame,
    *,
    pred_col: str = "prediction",
    target_col: str = TARGET_COLUMN,
    date_col: str = "date_cours",
    cfg: RecommendationConfig | None = None,
) -> dict[str, float]:
    """
    Rendements futurs BUY/NEUTRAL/SELL, spread BUY-SELL, hit ratio, turnover.
    Tau calibré en quantile des |pred| du même pli (mois-fin) — causal vs futur.
    """
    cfg = cfg or RecommendationConfig(
        benchmark_mode="zero_alpha",
        tau_method="fixed",
        tau_fixed=0.10,
    )
    # Sur 1 fold isolé : tau = quantile des |pred| mois-fin du fold (pas de futur)
    me = pred_df.loc[month_end_mask(pred_df[date_col])]
    if cfg.tau_method != "fixed" and not me.empty:
        cfg = RecommendationConfig(
            benchmark_mode=cfg.benchmark_mode,
            tau_method="train_abs_quantile",
            tau_quantile=cfg.tau_quantile,
            tau_fixed=cfg.tau_fixed,
        )
        ref = me[pred_col]
    else:
        ref = me[pred_col] if not me.empty else pred_df[pred_col]

    if cfg.tau_method == "fixed":
        recos = generate_recommendations(
            pred_df,
            cfg=cfg,
            rebalance_only=True,
            pred_col=pred_col,
            date_col=date_col,
            target_col=target_col,
        )
    else:
        recos = generate_recommendations(
            pred_df,
            train_predictions_for_tau=ref,
            cfg=cfg,
            rebalance_only=True,
            pred_col=pred_col,
            date_col=date_col,
            target_col=target_col,
        )

    if recos.empty or "realized_alpha" not in recos.columns:
        return {
            "ret_BUY": float("nan"),
            "ret_NEUTRAL": float("nan"),
            "ret_SELL": float("nan"),
            "spread_BUY_SELL": float("nan"),
            "hit_ratio": float("nan"),
            "turnover": float("nan"),
        }

    means = recos.groupby("recommendation")["realized_alpha"].mean()
    buy = float(means.get("BUY", np.nan))
    neu = float(means.get("NEUTRAL", np.nan))
    sell = float(means.get("SELL", np.nan))

    # Hit ratio sélection : BUY avec alpha>0 + SELL avec alpha<0
    buy_df = recos[recos["recommendation"] == "BUY"]
    sell_df = recos[recos["recommendation"] == "SELL"]
    hits: list[float] = []
    if len(buy_df):
        hits.extend((buy_df["realized_alpha"] > 0).astype(float).tolist())
    if len(sell_df):
        hits.extend((sell_df["realized_alpha"] < 0).astype(float).tolist())
    hit = float(np.mean(hits)) if hits else float("nan")

    return {
        "ret_BUY": buy,
        "ret_NEUTRAL": neu,
        "ret_SELL": sell,
        "spread_BUY_SELL": buy - sell
        if np.isfinite(buy) and np.isfinite(sell)
        else float("nan"),
        "hit_ratio": hit,
        "turnover": _reco_turnover(recos),
    }


def metrics_to_long(
    *,
    fold_id: int,
    model_id: str,
    level1: dict[str, float],
    level2: dict[str, float],
    groupe: str | None = None,
) -> list[dict[str, Any]]:
    """Format long : fold_id, model_id, groupe, métrique, valeur."""
    g = groupe or MODEL_GROUP.get(model_id, "unknown")
    rows: list[dict[str, Any]] = []
    for metric, value in {**level1, **level2}.items():
        rows.append(
            {
                "fold_id": int(fold_id),
                "model_id": model_id,
                "groupe": g,
                "metrique": metric,
                "valeur": float(value) if value is not None else float("nan"),
            }
        )
    return rows


def run_omnibus_test(
    results_df: pd.DataFrame,
    metric: str,
    *,
    fold_col: str = "fold_id",
    group_col: str = "groupe",
    metric_col: str = "metrique",
    value_col: str = "valeur",
) -> dict[str, Any]:
    """
    Test de Friedman sur la variable ``groupe`` (3 niveaux) pour une métrique.

    Agrège d'abord les modèles d'un même groupe (moyenne) par fold/bloc, puis
    compare les 3 groupes. Aucune comparaison pairwise automatique.
    """
    sub = results_df.loc[results_df[metric_col] == metric].copy()
    if sub.empty:
        return {
            "metric": metric,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n_blocks": 0,
            "n_groups": 0,
            "note": "aucune donnée pour cette métrique",
        }

    # Moyenne intra-groupe par fold
    agg = (
        sub.groupby([fold_col, group_col], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "valeur_groupe"})
    )
    wide = agg.pivot(index=fold_col, columns=group_col, values="valeur_groupe")
    # Ordre stable des 3 groupes
    order = [c for c in ("classique", "hybride", "hybride_regime") if c in wide.columns]
    if len(order) < 2:
        return {
            "metric": metric,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n_blocks": int(len(wide)),
            "n_groups": len(order),
            "note": "moins de 2 groupes disponibles",
        }
    mat = wide[order].dropna(how="any")
    n_blocks = int(len(mat))
    if n_blocks < 2:
        return {
            "metric": metric,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n_blocks": n_blocks,
            "n_groups": len(order),
            "groups": order,
            "note": "pas assez de blocs pour Friedman",
        }

    # scipy.stats.friedmanchisquare attend k arrays de longueur n
    arrays = [mat[c].to_numpy(dtype=float) for c in order]
    try:
        stat, pvalue = stats.friedmanchisquare(*arrays)
    except ValueError as exc:
        return {
            "metric": metric,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n_blocks": n_blocks,
            "n_groups": len(order),
            "groups": order,
            "note": str(exc),
        }

    return {
        "metric": metric,
        "stat": float(stat),
        "pvalue": float(pvalue),
        "n_blocks": n_blocks,
        "n_groups": len(order),
        "groups": order,
        "group_means": {c: float(mat[c].mean()) for c in order},
    }


def bootstrap_stationary_ci(
    results_df: pd.DataFrame,
    metric: str,
    model_id: str,
    *,
    block_size: float | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    fold_col: str = "fold_id",
    model_col: str = "model_id",
    metric_col: str = "metrique",
    value_col: str = "valeur",
) -> dict[str, Any]:
    """
    IC 95 % par bootstrap stationnaire (Politis–Romano) sur les folds overlapping.

    Longueur de bloc ~ Geometric(p) avec p = 1/block_size.
    Si ``block_size`` est None : heuristique ceil(n^{1/3}).
    """
    sub = results_df.loc[
        (results_df[metric_col] == metric) & (results_df[model_col] == model_id)
    ].copy()
    if sub.empty:
        return {
            "metric": metric,
            "model_id": model_id,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_folds": 0,
            "n_bootstrap": n_bootstrap,
        }

    series = (
        sub.sort_values(fold_col)[value_col]
        .to_numpy(dtype=float)
    )
    n = int(len(series))
    if n == 0:
        return {
            "metric": metric,
            "model_id": model_id,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_folds": 0,
            "n_bootstrap": n_bootstrap,
        }

    mean_obs = float(np.nanmean(series))
    if n == 1 or not np.isfinite(mean_obs):
        return {
            "metric": metric,
            "model_id": model_id,
            "mean": mean_obs,
            "ci_low": mean_obs,
            "ci_high": mean_obs,
            "n_folds": n,
            "n_bootstrap": n_bootstrap,
            "block_size": block_size,
        }

    bs = float(block_size) if block_size is not None else float(max(2, int(np.ceil(n ** (1 / 3)))))
    p = 1.0 / bs
    rng = np.random.default_rng(random_state)

    boots: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            # Longueur géométrique (moyenne = block_size)
            length = int(rng.geometric(p))
            for j in range(length):
                sample.append(float(series[(start + j) % n]))
                if len(sample) >= n:
                    break
        boots.append(float(np.nanmean(sample[:n])))

    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "model_id": model_id,
        "mean": mean_obs,
        "ci_low": lo,
        "ci_high": hi,
        "n_folds": n,
        "n_bootstrap": n_bootstrap,
        "block_size": bs,
        "alpha": alpha,
    }


def _wide_metric_by_block(
    results_df: pd.DataFrame,
    metric: str,
    *,
    fold_col: str = "fold_id",
    model_col: str = "model_id",
    metric_col: str = "metrique",
    value_col: str = "valeur",
) -> pd.DataFrame:
    sub = results_df.loc[results_df[metric_col] == metric].copy()
    wide = sub.pivot(index=fold_col, columns=model_col, values=value_col)
    return wide.sort_index()


def run_pairwise_test(
    results_df: pd.DataFrame,
    ref_model: str,
    alt_model: str,
    metric: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """
    Comparaison appariée sur blocs indépendants : Wilcoxon + DM (HLN).

    Pour rank_ic / spread : ``higher_is_better=True``.
    Pour rmse / turnover : ``higher_is_better=False``.
    """
    from experiments.factorial_hybrid_adapt.run_wf_dm_ic_blocks import diebold_mariano

    wide = _wide_metric_by_block(results_df, metric)
    if ref_model not in wide.columns or alt_model not in wide.columns:
        return {
            "ref_model": ref_model,
            "alt_model": alt_model,
            "metric": metric,
            "n_blocks": 0,
            "note": "modèle absent du panel",
        }

    mat = wide[[ref_model, alt_model]].dropna(how="any")
    ref = mat[ref_model].to_numpy(dtype=float)
    alt = mat[alt_model].to_numpy(dtype=float)
    n = int(len(mat))
    diff = alt - ref  # >0 si alt meilleure (higher_is_better)

    delta_mean = float(np.mean(diff))
    delta_median = float(np.median(diff))

    if n >= 3:
        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    else:
        w_stat, w_p = float("nan"), float("nan")

    # DM sur pertes : loss = -metric si higher_is_better
    if higher_is_better:
        loss_ref, loss_alt = -ref, -alt
    else:
        loss_ref, loss_alt = ref, alt
    dm = diebold_mariano(loss_ref, loss_alt, h=1)

    return {
        "pair": f"{alt_model}_vs_{ref_model}",
        "ref_model": ref_model,
        "alt_model": alt_model,
        "metric": metric,
        "n_blocks": n,
        "delta_mean": delta_mean,
        "delta_median": delta_median,
        "ref_mean": float(np.mean(ref)),
        "alt_mean": float(np.mean(alt)),
        "wilcoxon_stat": None if pd.isna(w_stat) else float(w_stat),
        "pvalue_wilcoxon": None if pd.isna(w_p) else float(w_p),
        "sig_5pct_wilcoxon": bool(np.isfinite(w_p) and w_p < 0.05),
        "dm_stat_hln": dm.get("dm_stat_hln"),
        "pvalue_dm_hln": dm.get("pvalue_hln"),
        "sig_5pct_dm_hln": bool(
            np.isfinite(dm.get("pvalue_hln", np.nan)) and dm["pvalue_hln"] < 0.05
        ),
        "interpretation": (
            f"delta>0 : {alt_model} meilleur que {ref_model} en moyenne sur {metric}"
        ),
    }


def run_pairwise_battery(
    results_df: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    metrics: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    """
    Batterie de tests pairwise.

    ``pairs`` : liste (ref, alt, label).
    ``metrics`` : liste (nom_métrique, higher_is_better).
    """
    if metrics is None:
        metrics = [
            ("rank_ic", True),
            ("ic_ir", True),
            ("spread_BUY_SELL", True),
            ("rmse", False),
            ("turnover", False),
        ]
    rows: list[dict[str, Any]] = []
    for ref, alt, label in pairs:
        for met, hib in metrics:
            row = run_pairwise_test(results_df, ref, alt, met, higher_is_better=hib)
            row["label"] = label
            rows.append(row)
    return pd.DataFrame(rows)
