"""
Sous-module D — Optimisation du portefeuille (pymoo NSGA-III).

Prend les actions labellisées TOP et trouve les poids optimaux
en maximisant simultanément 3 objectifs concurrents :
1. Maximiser l'alpha attendu (score_final)
2. Minimiser le CVaR 95 %
3. Maximiser la liquidité du portefeuille

Contraintes :
- Σ w = 1
- w_min = 2 %, w_max = 10 %
- CVaR_portefeuille ≤ CVaR_MASI × 0.9

Output : frontière de Pareto → 3 portefeuilles
- P_agressif  : maximise l'alpha
- P_equilibre : point genou
- P_defensif  : minimise le CVaR
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD, RANDOM_STATE
from bvc_recommender.models.liquidity_filter import LABEL_TOP

logger = logging.getLogger(__name__)

PORTFOLIO_MIN_WEIGHT = 0.02
PORTFOLIO_MAX_WEIGHT = 0.10
CVAR_ALPHA = 0.05
CVAR_MASI_RATIO = 0.9
NSGA_POP_SIZE = 92
NSGA_GENERATIONS = 80
NSGA_N_PARTITIONS = 8  # → 45 directions ; pop_size ≥ 45
RETURNS_LOOKBACK_DAYS = 504  # ~2 ans de séances


def _daily_returns(cours: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    c = cours[cours["ticker"].isin(tickers)].copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    c = c.dropna(subset=["date_cours", price_col])
    wide = c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last")
    rets = wide.pct_change(fill_method=None).dropna(how="all")
    if len(rets) > RETURNS_LOOKBACK_DAYS:
        rets = rets.iloc[-RETURNS_LOOKBACK_DAYS:]
    return rets


def _masi_returns(indices: pd.DataFrame) -> pd.Series:
    df = indices.copy()
    code_col = "code_index" if "code_index" in df.columns else "code"
    date_col = "date_index" if "date_index" in df.columns else "date"
    val_col = "valeur_index" if "valeur_index" in df.columns else "valeur"
    masi = df[df[code_col].astype(str).str.upper() == "MASI"].copy()
    masi[date_col] = pd.to_datetime(masi[date_col], errors="coerce")
    masi[val_col] = pd.to_numeric(masi[val_col], errors="coerce")
    masi = masi.dropna(subset=[date_col, val_col]).sort_values(date_col)
    prices = masi.drop_duplicates(date_col, keep="last").set_index(date_col)[val_col]
    rets = prices.pct_change(fill_method=None).dropna()
    if len(rets) > RETURNS_LOOKBACK_DAYS:
        rets = rets.iloc[-RETURNS_LOOKBACK_DAYS:]
    return rets


def compute_cvar(returns: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """CVaR (expected shortfall) sur un vecteur de rendements."""
    if len(returns) == 0:
        return 0.0
    q = np.quantile(returns, alpha)
    tail = returns[returns <= q]
    return float(-tail.mean()) if len(tail) else float(-q)


def portfolio_cvar(weights: np.ndarray, returns_matrix: np.ndarray) -> float:
    port_rets = returns_matrix @ weights
    return compute_cvar(port_rets)


def project_weights(
    weights: np.ndarray,
    *,
    w_min: float = PORTFOLIO_MIN_WEIGHT,
    w_max: float = PORTFOLIO_MAX_WEIGHT,
) -> np.ndarray:
    """Projette les poids sur le simplexe avec bornes [w_min, w_max]."""
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    if n == 0:
        return w
    # Si les bornes sont incompatibles avec Σw=1, assouplir le min
    if n * w_min > 1.0 + 1e-9:
        w_min = 1.0 / n
    if n * w_max < 1.0 - 1e-9:
        w_max = 1.0 / n

    w = np.clip(w, w_min, w_max)
    for _ in range(50):
        total = w.sum()
        if abs(total - 1.0) < 1e-9:
            break
        if total <= 0:
            w = np.full(n, 1.0 / n)
            break
        w = w / total
        w = np.clip(w, w_min, w_max)
    # Correction finale de masse
    deficit = 1.0 - w.sum()
    if abs(deficit) > 1e-9:
        free = np.where(deficit > 0, w_max - w, w - w_min)
        if free.sum() > 0:
            w = w + deficit * (free / free.sum())
        w = np.clip(w, w_min, w_max)
        w = w / w.sum()
    return w


def _knee_point_index(objectives: np.ndarray) -> int:
    """Point genou sur la frontière de Pareto (distance normalisée à l'utopie)."""
    obj = objectives.copy()
    mins = obj.min(axis=0)
    maxs = obj.max(axis=0)
    span = np.where(maxs - mins > 0, maxs - mins, 1.0)
    norm = (obj - mins) / span
    # f1=−alpha, f2=CVaR, f3=−liq  (tous à minimiser déjà dans F)
    dist = np.linalg.norm(norm, axis=1)
    return int(np.argmin(dist))


def _estimate_min_cvar(
    ret_mat: np.ndarray,
    *,
    w_min: float,
    w_max: float,
) -> tuple[float, np.ndarray]:
    """CVaR minimal atteignable sous bornes de poids (SLSQP)."""
    from scipy.optimize import minimize as scipy_minimize

    n = ret_mat.shape[1]
    w0 = np.full(n, 1.0 / n)

    def obj(w: np.ndarray) -> float:
        return portfolio_cvar(project_weights(w, w_min=w_min, w_max=w_max), ret_mat)

    res = scipy_minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(w_min, w_max)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w_star = project_weights(res.x if res.success else w0, w_min=w_min, w_max=w_max)
    return portfolio_cvar(w_star, ret_mat), w_star


def _resolve_cvar_limit(
    cvar_masi: float,
    min_cvar: float,
    *,
    target_ratio: float = CVAR_MASI_RATIO,
    slack: float = 1.02,
) -> tuple[float, float, bool]:
    """
    Retourne (limite_cible, limite_effective, relaxed).

    Si 0.9×CVaR_MASI est inatteignable sous les bornes de poids, assouplit
    jusqu'à min_cvar × slack (plus petite région faisable documentée).
    """
    target = cvar_masi * target_ratio
    if min_cvar <= target + 1e-12:
        return target, target, False
    effective = min_cvar * slack
    return target, effective, True


def optimize_portfolios(
    ranked: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    label_filter: str = LABEL_TOP,
    pop_size: int = NSGA_POP_SIZE,
    n_gen: int = NSGA_GENERATIONS,
) -> dict[str, Any]:
    """
    Optimise les poids sur les actions ``label_filter`` via NSGA-III.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    candidates = ranked[ranked["label"] == label_filter].copy()
    if candidates.empty:
        candidates = ranked.nlargest(min(20, len(ranked)), "score_final")

    tickers = candidates["ticker"].tolist()
    n = len(tickers)
    if n < 2:
        raise ValueError(f"Pas assez de candidats pour optimiser ({n} tickers).")

    # Ajuster w_min si trop de titres TOP (Σ w_min ≤ 1)
    w_min = PORTFOLIO_MIN_WEIGHT
    if n * w_min > 1.0:
        w_min = round(1.0 / n, 4)
        logger.warning(
            "w_min assoupli à %.2f%% (%s titres TOP — contrainte Σw_min≤1)",
            w_min * 100,
            n,
        )

    ret_df = _daily_returns(cours, tickers).dropna(axis=1, how="all")
    common = [t for t in tickers if t in ret_df.columns and ret_df[t].notna().sum() > 60]
    if len(common) < 2:
        raise ValueError("Historique de rendements insuffisant pour l'optimisation.")
    tickers = common
    n = len(tickers)
    if n * w_min > 1.0:
        w_min = round(1.0 / n, 4)
    ret_mat = ret_df[tickers].fillna(0.0).values

    expected_alpha = candidates.set_index("ticker").loc[tickers, "score_final"].fillna(0).values
    liquidity = candidates.set_index("ticker").loc[tickers, "vmq_20j"].fillna(0).values
    liquidity_norm = liquidity / max(float(liquidity.max()), float(LIQUIDITY_VMQ_THRESHOLD_MAD))

    masi_ret = _masi_returns(indices)
    cvar_masi = compute_cvar(masi_ret.values)
    min_cvar, w_min_cvar = _estimate_min_cvar(ret_mat, w_min=w_min, w_max=PORTFOLIO_MAX_WEIGHT)
    cvar_limit_target, cvar_limit, constraint_relaxed = _resolve_cvar_limit(
        cvar_masi, min_cvar
    )
    if constraint_relaxed:
        logger.warning(
            "Contrainte CVaR %.2f×MASI (%.4f) inatteignable "
            "(min réalisable=%.4f ≈ %.2f×MASI) — limite effective=%.4f",
            CVAR_MASI_RATIO,
            cvar_limit_target,
            min_cvar,
            min_cvar / cvar_masi if cvar_masi > 0 else float("nan"),
            cvar_limit,
        )

    class PortfolioProblem(Problem):
        def __init__(self, limit: float) -> None:
            super().__init__(
                n_var=n,
                n_obj=3,
                n_ieq_constr=1,
                xl=np.full(n, w_min),
                xu=np.full(n, PORTFOLIO_MAX_WEIGHT),
            )
            self._limit = limit

        def _evaluate(self, x, out, *args, **kwargs) -> None:
            f1, f2, f3, g1 = [], [], [], []
            for raw in x:
                w = project_weights(raw, w_min=w_min, w_max=PORTFOLIO_MAX_WEIGHT)
                alpha = float(np.dot(w, expected_alpha))
                cvar = portfolio_cvar(w, ret_mat)
                liq = float(np.dot(w, liquidity_norm))
                f1.append(-alpha)  # maximiser alpha → minimiser -alpha
                f2.append(cvar)  # minimiser CVaR
                f3.append(-liq)  # maximiser liquidité → minimiser -liq
                g1.append(cvar - self._limit)  # ≤ 0
            out["F"] = np.column_stack([f1, f2, f3])
            out["G"] = np.array(g1).reshape(-1, 1)

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=NSGA_N_PARTITIONS)
    n_ref = len(ref_dirs)
    effective_pop = max(pop_size, n_ref)

    # Population initiale : min-CVaR + max-alpha + équipondéré + bruit
    rng = np.random.default_rng(RANDOM_STATE)
    w_alpha = project_weights(
        expected_alpha / max(expected_alpha.sum(), 1e-12),
        w_min=w_min,
        w_max=PORTFOLIO_MAX_WEIGHT,
    )
    w_liq = project_weights(
        liquidity_norm / max(liquidity_norm.sum(), 1e-12),
        w_min=w_min,
        w_max=PORTFOLIO_MAX_WEIGHT,
    )
    w_eq0 = np.full(n, 1.0 / n)
    seeds = [w_min_cvar, w_alpha, w_liq, w_eq0]
    init = np.vstack(
        seeds
        + [
            project_weights(
                s + rng.normal(0, 0.02, size=n),
                w_min=w_min,
                w_max=PORTFOLIO_MAX_WEIGHT,
            )
            for s in seeds
            for _ in range(max(1, effective_pop // (4 * len(seeds))) )
        ]
    )
    if len(init) < effective_pop:
        extra = rng.uniform(w_min, PORTFOLIO_MAX_WEIGHT, size=(effective_pop - len(init), n))
        extra_proj = np.vstack(
            [project_weights(row, w_min=w_min, w_max=PORTFOLIO_MAX_WEIGHT) for row in extra]
        )
        init = np.vstack([init, extra_proj])
    init = init[:effective_pop]

    from pymoo.core.sampling import Sampling

    class SeededSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return init[:n_samples].copy()

    algorithm = NSGA3(
        pop_size=effective_pop,
        ref_dirs=ref_dirs,
        sampling=SeededSampling(),
    )
    problem = PortfolioProblem(cvar_limit)

    logger.info(
        "NSGA-III — %s titres TOP | pop=%s | gén=%s | CVaR_MASI=%.4f | "
        "limite_cible=%.4f | limite_effective=%.4f",
        n,
        effective_pop,
        n_gen,
        cvar_masi,
        cvar_limit_target,
        cvar_limit,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=RANDOM_STATE,
        verbose=False,
    )

    pareto_x = res.X
    pareto_f = res.F
    if pareto_x is None or len(np.atleast_2d(pareto_x)) == 0:
        # Repli : optimisation sans contrainte CVaR, puis filtrage soft
        logger.warning(
            "Pareto vide sous contrainte CVaR — relance sans contrainte dure "
            "(sélection des 3 profils sur la frontière libre)."
        )
        problem_free = PortfolioProblem(limit=1e9)
        algorithm_free = NSGA3(
            pop_size=effective_pop,
            ref_dirs=ref_dirs,
            sampling=SeededSampling(),
        )
        res = minimize(
            problem_free,
            algorithm_free,
            ("n_gen", n_gen),
            seed=RANDOM_STATE,
            verbose=False,
        )
        pareto_x = res.X
        pareto_f = res.F

    if pareto_x is None or len(np.atleast_2d(pareto_x)) == 0:
        logger.warning("Frontière de Pareto vide — portefeuilles équipondérés de secours.")
        w_eq = np.full(n, 1.0 / n)
        return _format_portfolios(
            tickers,
            w_eq,
            w_eq,
            w_eq,
            expected_alpha,
            liquidity_norm,
            ret_mat,
            cvar_masi,
            cvar_limit,
            n_pareto=0,
            cvar_limit_target=cvar_limit_target,
            min_cvar=min_cvar,
            constraint_relaxed=constraint_relaxed,
            w_min_used=w_min,
        )

    pareto_x = np.atleast_2d(pareto_x)
    pareto_f = np.atleast_2d(pareto_f)
    pareto_w = np.vstack(
        [project_weights(w, w_min=w_min, w_max=PORTFOLIO_MAX_WEIGHT) for w in pareto_x]
    )

    idx_aggressive = int(np.argmin(pareto_f[:, 0]))  # max alpha
    idx_defensive = int(np.argmin(pareto_f[:, 1]))  # min CVaR
    idx_equilibrium = _knee_point_index(pareto_f)

    w_agg = pareto_w[idx_aggressive]
    w_eq = pareto_w[idx_equilibrium]
    w_def = pareto_w[idx_defensive]

    logger.info("NSGA-III : %s solutions Pareto retenues", len(pareto_w))
    return _format_portfolios(
        tickers,
        w_agg,
        w_eq,
        w_def,
        expected_alpha,
        liquidity_norm,
        ret_mat,
        cvar_masi,
        cvar_limit,
        n_pareto=len(pareto_w),
        cvar_limit_target=cvar_limit_target,
        min_cvar=min_cvar,
        constraint_relaxed=constraint_relaxed,
        w_min_used=w_min,
    )


def _format_portfolios(
    tickers: list[str],
    w_agg: np.ndarray,
    w_eq: np.ndarray,
    w_def: np.ndarray,
    expected_alpha: np.ndarray,
    liquidity_norm: np.ndarray,
    ret_mat: np.ndarray,
    cvar_masi: float,
    cvar_limit: float,
    *,
    n_pareto: int,
    cvar_limit_target: float | None = None,
    min_cvar: float | None = None,
    constraint_relaxed: bool = False,
    w_min_used: float = PORTFOLIO_MIN_WEIGHT,
) -> dict[str, Any]:
    if cvar_limit_target is None:
        cvar_limit_target = cvar_limit

    def _pack(name: str, weights: np.ndarray) -> dict[str, Any]:
        w = project_weights(weights, w_min=w_min_used, w_max=PORTFOLIO_MAX_WEIGHT)
        alpha = float(np.dot(w, expected_alpha))
        cvar = portfolio_cvar(w, ret_mat)
        liq = float(np.dot(w, liquidity_norm))
        df = pd.DataFrame({"ticker": tickers, "weight": w})
        df = df[df["weight"] >= 0.005].sort_values("weight", ascending=False)
        df["weight_pct"] = (df["weight"] * 100).round(2)
        return {
            "name": name,
            "tickers": df["ticker"].tolist(),
            "weights": df["weight"].round(4).tolist(),
            "weights_pct": df["weight_pct"].tolist(),
            "n_positions": len(df),
            "expected_alpha": round(alpha, 6),
            "cvar_95": round(cvar, 6),
            "liquidity_score": round(liq, 4),
            "cvar_vs_limit": round(cvar / cvar_limit, 4) if cvar_limit > 0 else None,
            "cvar_ok": bool(cvar <= cvar_limit + 1e-9),
        }

    result = {
        "method": "NSGA-III (pymoo)",
        "objectives": [
            "max expected alpha (score_final)",
            "min CVaR 95%",
            "max portfolio liquidity (VMQ)",
        ],
        "constraints": {
            "sum_weights": 1.0,
            "w_min": w_min_used,
            "w_max": PORTFOLIO_MAX_WEIGHT,
            "cvar_masi_ratio_target": CVAR_MASI_RATIO,
            "cvar_constraint_relaxed": constraint_relaxed,
        },
        "cvar_masi": round(cvar_masi, 6),
        "cvar_limit_target": round(cvar_limit_target, 6),
        "cvar_limit": round(cvar_limit, 6),
        "min_cvar_achievable": round(min_cvar, 6) if min_cvar is not None else None,
        "n_pareto": n_pareto,
        "candidates": len(tickers),
        "P_agressif": _pack("Agressif", w_agg),
        "P_equilibre": _pack("Équilibré", w_eq),
        "P_defensif": _pack("Défensif", w_def),
    }
    return result

