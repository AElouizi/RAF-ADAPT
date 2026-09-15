"""
Étage 2 — Allocation NSGA-III sur l'univers des recommandations (plus de Top 25).

## Fenêtre (commune aux variantes)
- ``EVAL_START_MONTH`` → ``EVAL_END_MONTH`` (2015-06 → 2025-06 ; recos dès 2018-07)
- Exclusion SELL (si sell_mode=exclude)
- VMQ as-of t uniquement ; aucune imputation ; pas d'info future

## Variantes liquidité (``liquidity_mode``)
- ``pareto`` (protocole principal validé — plateforme RAF-ADAPT) :
    pas de filtre VMQ ; NSGA 3 obj (alpha, CVaR, max L=sigmoid(VMQ))
- ``eligibility`` :
    filtre VMQ_20j >= 500k puis NSGA 2 obj (alpha, CVaR)
- ``none`` :
    pas de filtre VMQ ; NSGA 2 obj (alpha, CVaR) — liquidité ignorée
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD, RANDOM_STATE
from bvc_recommender.models.liquidity_filter import sigmoid_liquidity_factor
from bvc_recommender.models.portfolio_optimizer import (
    CVAR_MASI_RATIO,
    NSGA_N_PARTITIONS,
    PORTFOLIO_MAX_WEIGHT,
    RETURNS_LOOKBACK_DAYS,
    _daily_returns,
    _estimate_min_cvar,
    _knee_point_index,
    _masi_returns,
    _resolve_cvar_limit,
    compute_cvar,
    portfolio_cvar,
    project_weights,
)
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.recommendations import CELL_META

logger = logging.getLogger(__name__)

SellMode = Literal["exclude", "penalize"]
SelectionRule = Literal[
    "knee",
    "ideal_distance",
    "max_alpha",
    "min_cvar",
    "max_alpha_constrained",
]
LiquidityMode = Literal["eligibility", "pareto", "none"]

# Params walk-forward (alignés étape 5 factorielle)
NSGA_POP_STAGE2 = 36
NSGA_GEN_STAGE2 = 30

# Fenêtre Stage 2 : features dès 2015-06 ; premier test WF = 2018-07 (36 mois de train).
EVAL_START_MONTH = "2015-06"
EVAL_END_MONTH = "2025-06"

# L_min méthodologique = sigmoid(500k MAD)
DEFAULT_L_MIN = float(
    sigmoid_liquidity_factor(pd.Series([float(LIQUIDITY_VMQ_THRESHOLD_MAD)])).iloc[0]
)


@dataclass(frozen=True)
class Stage2Config:
    """Hyperparamètres allocation étage 2 (aucun look-ahead)."""

    sell_mode: SellMode = "exclude"
    pref_BUY: float = 1.0
    pref_NEUTRAL: float = 0.5
    pref_SELL: float = 0.0  # utilisé si sell_mode=penalize
    w_min: float = 0.0  # sparse long-only
    w_max: float = PORTFOLIO_MAX_WEIGHT  # 10 %
    w_max_SELL: float = 0.02  # si penalize
    # Liquidité : eligibility | pareto | none (voir module docstring)
    liquidity_mode: LiquidityMode = "pareto"
    # Filtre VMQ >= min_vmq même en mode pareto (3 obj.). NaN exclus (pas d'imputation).
    apply_vmq_filter: bool = False
    min_vmq: float = float(LIQUIDITY_VMQ_THRESHOLD_MAD)
    l_min: float = DEFAULT_L_MIN  # sigmoid(min_vmq) ; documentation / contrôles
    max_turnover: float | None = None
    selection_rule: SelectionRule = "knee"
    pop_size: int = NSGA_POP_STAGE2
    n_gen: int = NSGA_GEN_STAGE2
    weight_report_floor: float = 1e-4
    eval_start_month: str = EVAL_START_MONTH
    eval_end_month: str = EVAL_END_MONTH

    @property
    def enforce_liquidity_eligibility(self) -> bool:
        return self.liquidity_mode == "eligibility" or bool(self.apply_vmq_filter)


def preference_multiplier(recommendation: str, cfg: Stage2Config) -> float:
    rec = str(recommendation).upper()
    if rec == "BUY":
        return float(cfg.pref_BUY)
    if rec == "NEUTRAL":
        return float(cfg.pref_NEUTRAL)
    if rec == "SELL":
        return float(cfg.pref_SELL)
    return float(cfg.pref_NEUTRAL)


def prepare_universe(
    month_panel: pd.DataFrame,
    cfg: Stage2Config,
) -> pd.DataFrame:
    """
    Univers NSGA à t :
      1) VMQ_20j as-of t (déjà dans le panel) → L_i = sigmoid(VMQ)
      2) si liquidity_mode=eligibility : filtre VMQ >= min_vmq
      3) exclusion SELL (si sell_mode=exclude)
    Aucune imputation de VMQ. Pas de N fixe.
    """
    df = month_panel.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors="coerce")
    df["conviction_score"] = pd.to_numeric(
        df.get("conviction_score", df["predicted_return"]), errors="coerce"
    )
    df["liquidity_vmq_20j"] = pd.to_numeric(df["liquidity_vmq_20j"], errors="coerce")
    df["recommendation"] = df["recommendation"].astype(str).str.upper()
    df = df.dropna(subset=["ticker", "predicted_return", "recommendation"])

    n0 = len(df)
    # L : NaN VMQ → 0 uniquement pour le facteur (titres ensuite exclus si filtre).
    df["L_sigmoid"] = sigmoid_liquidity_factor(df["liquidity_vmq_20j"].fillna(0.0))
    n_liq_count = int(
        (df["liquidity_vmq_20j"].notna() & (df["liquidity_vmq_20j"] >= float(cfg.min_vmq))).sum()
    )

    if cfg.enforce_liquidity_eligibility:
        # Aucune imputation : VMQ manquant ⇒ non éligible
        df = df[
            df["liquidity_vmq_20j"].notna()
            & (df["liquidity_vmq_20j"] >= float(cfg.min_vmq))
        ].copy()
    n_after_liq = len(df)

    if cfg.sell_mode == "exclude":
        df = df[df["recommendation"] != "SELL"].copy()
    elif cfg.sell_mode != "penalize":
        raise ValueError(f"sell_mode inconnu : {cfg.sell_mode}")
    n_nsga = len(df)

    logger.info(
        "Univers t [%s] : n0=%s → n_VMQ>=seuil=%s → après_liq=%s → après SELL(%s)=%s",
        cfg.liquidity_mode,
        n0,
        n_liq_count,
        n_after_liq,
        cfg.sell_mode,
        n_nsga,
    )

    df = df.drop_duplicates("ticker", keep="first")
    df["pref_mult"] = df["recommendation"].map(lambda r: preference_multiplier(r, cfg))
    df["score_opt"] = df["predicted_return"] * df["pref_mult"]
    df["w_max_i"] = np.where(
        (df["recommendation"] == "SELL") & (cfg.sell_mode == "penalize"),
        cfg.w_max_SELL,
        cfg.w_max,
    )
    df.attrs["n_univ_raw"] = n0
    df.attrs["n_liquid"] = n_liq_count
    df.attrs["n_after_liq_filter"] = n_after_liq
    df.attrs["n_nsga"] = len(df)
    return df.reset_index(drop=True)


def _ideal_distance_index(objectives: np.ndarray) -> int:
    """Distance euclidienne normalisée à l'utopie (équivalent knee existant)."""
    return _knee_point_index(objectives)


def select_max_alpha_constrained_index(
    alpha: np.ndarray,
    cvar: np.ndarray,
    liquidity: np.ndarray,
    *,
    cvar_limit: float,
    l_min: float,
    cvar_slack: float = 1.02,
) -> tuple[int, str, int]:
    """
    Max Alpha sur le front, sous contraintes protocole (non calibrées sur l'OOS).

    Contraintes pré-spécifiées :
      - CVaR ≤ cvar_limit  (cvar_limit = 0.9 × CVaR_MASI, déjà utilisé dans NSGA)
      - L pondéré ≥ l_min  (l_min = sigmoid(VMQ=500k MAD), déjà dans Stage2Config)

    Hiérarchie si l'ensemble admissible est vide (même slack CVaR que
    ``_resolve_cvar_limit``, sans toucher l_min) :
      1) CVaR ≤ limit ET L ≥ l_min
      2) CVaR ≤ limit  (L-min inatteignable ce mois)
      3) CVaR ≤ min(front)×slack ET L ≥ l_min
      4) CVaR ≤ min(front)×slack
      5) front entier
    """
    a = np.asarray(alpha, dtype=float)
    c = np.asarray(cvar, dtype=float)
    L = np.asarray(liquidity, dtype=float)
    n = len(a)
    if n == 0:
        raise ValueError("front Pareto vide")
    if len(c) != n or len(L) != n:
        raise ValueError("alpha / cvar / liquidity de longueurs distinctes")

    ok_cvar = c <= float(cvar_limit) + 1e-9
    ok_liq = L >= float(l_min) - 1e-12
    min_c = float(np.nanmin(c)) if np.isfinite(c).any() else float(cvar_limit)
    ok_cvar_relaxed = c <= (min_c * float(cvar_slack) + 1e-9)

    stages = (
        ("cvar_and_L", ok_cvar & ok_liq),
        ("cvar_only", ok_cvar),
        ("cvar_relaxed_and_L", ok_cvar_relaxed & ok_liq),
        ("cvar_relaxed_only", ok_cvar_relaxed),
        ("unconstrained", np.ones(n, dtype=bool)),
    )

    def _best(mask: np.ndarray) -> int:
        idx = np.flatnonzero(mask)
        # tie-break déterministe : max alpha, puis min CVaR, puis max L, puis sol_id
        order = np.lexsort((idx, -L[idx], c[idx], -a[idx]))
        return int(idx[order[0]])

    for name, mask in stages:
        if np.any(mask):
            return _best(mask), name, int(mask.sum())
    return 0, "unconstrained", n


def _select_pareto_index(
    pareto_f: np.ndarray,
    rule: SelectionRule,
    *,
    cvar_limit: float | None = None,
    l_min: float | None = None,
) -> int:
    if rule in ("knee", "ideal_distance"):
        return _ideal_distance_index(pareto_f)
    if rule == "max_alpha":
        return int(np.argmin(pareto_f[:, 0]))  # F1 = -alpha
    if rule == "min_cvar":
        return int(np.argmin(pareto_f[:, 1]))
    if rule == "max_alpha_constrained":
        if cvar_limit is None or l_min is None:
            raise ValueError("max_alpha_constrained requiert cvar_limit et l_min")
        alpha = -np.asarray(pareto_f[:, 0], dtype=float)
        cvar = np.asarray(pareto_f[:, 1], dtype=float)
        liq = (
            -np.asarray(pareto_f[:, 2], dtype=float)
            if pareto_f.shape[1] > 2
            else np.ones(len(pareto_f), dtype=float)
        )
        idx, _stage, _n = select_max_alpha_constrained_index(
            alpha, cvar, liq, cvar_limit=cvar_limit, l_min=l_min
        )
        return idx
    raise ValueError(f"selection_rule inconnu : {rule}")


def _project_with_caps(
    weights: np.ndarray,
    *,
    w_min: float,
    w_max_vec: np.ndarray,
) -> np.ndarray:
    """Projection simplexe avec plafonds hétérogènes par titre."""
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    if n == 0:
        return w
    caps = np.asarray(w_max_vec, dtype=float)
    lo = float(w_min)
    if n * lo > 1.0 + 1e-9:
        lo = 1.0 / n
    if caps.sum() < 1.0 - 1e-9:
        # Caps incompatibles : assouplir proportionnellement
        caps = caps * (1.0 / max(caps.sum(), 1e-12))
    w = np.clip(w, lo, caps)
    for _ in range(80):
        total = w.sum()
        if abs(total - 1.0) < 1e-9:
            break
        if total <= 0:
            w = np.minimum(caps, np.full(n, 1.0 / n))
            w = w / w.sum()
            break
        w = w / total
        w = np.clip(w, lo, caps)
    deficit = 1.0 - w.sum()
    if abs(deficit) > 1e-9:
        free = np.where(deficit > 0, caps - w, w - lo)
        if free.sum() > 0:
            w = w + deficit * (free / free.sum())
        w = np.clip(w, lo, caps)
        if w.sum() > 0:
            w = w / w.sum()
            w = np.clip(w, lo, caps)
            if w.sum() > 0:
                w = w / w.sum()
    return w


def optimize_universe(
    universe: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    cfg: Stage2Config,
    prev_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    NSGA-III sur l'univers déjà filtré selon ``cfg.liquidity_mode`` :
      - eligibility / none : 2 obj (max score_opt, min CVaR)
      - pareto : 3 obj (+ max L=sigmoid(VMQ) pondéré)
    Historiques strictement ≤ as_of.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    as_of = pd.Timestamp(as_of)
    use_liq_obj = cfg.liquidity_mode == "pareto"
    n_obj = 3 if use_liq_obj else 2

    tickers0 = universe["ticker"].astype(str).tolist()
    if len(tickers0) < 2:
        raise ValueError(f"Univers trop petit ({len(tickers0)}) à {as_of.date()}")

    cours_asof = cours.copy()
    cours_asof["date_cours"] = pd.to_datetime(cours_asof["date_cours"], errors="coerce")
    cours_asof = cours_asof[cours_asof["date_cours"] <= as_of]

    idx_date = "date_index" if "date_index" in indices.columns else "date"
    indices_asof = indices.copy()
    indices_asof[idx_date] = pd.to_datetime(indices_asof[idx_date], errors="coerce")
    indices_asof = indices_asof[indices_asof[idx_date] <= as_of]

    ret_df = _daily_returns(cours_asof, tickers0).dropna(axis=1, how="all")
    common = [t for t in tickers0 if t in ret_df.columns and ret_df[t].notna().sum() > 60]
    if len(common) < 2:
        raise ValueError("Historique de rendements insuffisant (as-of).")

    uni = universe.set_index("ticker").loc[common].copy()
    tickers = common
    n = len(tickers)
    ret_mat = ret_df[tickers].fillna(0.0).values

    score = uni["score_opt"].fillna(0.0).to_numpy(dtype=float)
    raw_pred = uni["predicted_return"].fillna(0.0).to_numpy(dtype=float)
    liq = uni["liquidity_vmq_20j"].fillna(0.0).to_numpy(dtype=float)
    L_sig = (
        uni["L_sigmoid"].fillna(0.0).to_numpy(dtype=float)
        if "L_sigmoid" in uni.columns
        else sigmoid_liquidity_factor(pd.Series(liq)).to_numpy(dtype=float)
    )
    recs = uni["recommendation"].astype(str).tolist()
    conv = uni["conviction_score"].fillna(0.0).to_numpy(dtype=float)
    w_max_vec = uni["w_max_i"].to_numpy(dtype=float)

    w_min = float(cfg.w_min)
    if n * w_min > 1.0:
        w_min = 1.0 / n

    masi_ret = _masi_returns(indices_asof)
    cvar_masi = compute_cvar(masi_ret.values)
    w_max_hom = float(np.median(w_max_vec))
    min_cvar, w_min_cvar = _estimate_min_cvar(ret_mat, w_min=w_min, w_max=w_max_hom)
    w_min_cvar = _project_with_caps(w_min_cvar, w_min=w_min, w_max_vec=w_max_vec)
    cvar_limit_target, cvar_limit, constraint_relaxed = _resolve_cvar_limit(
        cvar_masi, min_cvar
    )

    prev_w = np.zeros(n)
    if prev_weights:
        for i, t in enumerate(tickers):
            prev_w[i] = float(prev_weights.get(t, 0.0))
        if prev_w.sum() > 0:
            prev_w = prev_w / prev_w.sum()

    n_ieq = 1 + (1 if cfg.max_turnover is not None and prev_w.sum() > 0 else 0)

    class PortfolioProblem(Problem):
        def __init__(self, limit: float) -> None:
            super().__init__(
                n_var=n,
                n_obj=n_obj,
                n_ieq_constr=n_ieq,
                xl=np.full(n, w_min),
                xu=w_max_vec.copy(),
            )
            self._limit = limit

        def _evaluate(self, x, out, *args, **kwargs) -> None:
            f_rows = []
            g_rows = []
            for raw in x:
                w = _project_with_caps(raw, w_min=w_min, w_max_vec=w_max_vec)
                alpha = float(np.dot(w, score))
                cvar = portfolio_cvar(w, ret_mat)
                row = [-alpha, cvar]
                if use_liq_obj:
                    row.append(-float(np.dot(w, L_sig)))  # max L → min -L
                f_rows.append(row)
                gs = [cvar - self._limit]
                if n_ieq == 2:
                    to = 0.5 * float(np.abs(w - prev_w).sum())
                    gs.append(to - float(cfg.max_turnover))
                g_rows.append(gs)
            out["F"] = np.asarray(f_rows, dtype=float)
            out["G"] = np.asarray(g_rows, dtype=float)

    ref_dirs = get_reference_directions(
        "das-dennis", n_obj, n_partitions=NSGA_N_PARTITIONS
    )
    effective_pop = max(int(cfg.pop_size), len(ref_dirs))
    rng = np.random.default_rng(RANDOM_STATE)

    def _seed_from(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        v = np.maximum(v, 0)
        if v.sum() <= 0:
            v = np.ones(n)
        return _project_with_caps(v, w_min=w_min, w_max_vec=w_max_vec)

    seeds = [
        w_min_cvar,
        _seed_from(np.maximum(score, 0)),
        _seed_from(np.ones(n)),
    ]
    if use_liq_obj:
        seeds.append(_seed_from(np.maximum(L_sig, 0)))
    buy_mask = np.array([r == "BUY" for r in recs], dtype=bool)
    if buy_mask.any():
        v = np.where(buy_mask, np.maximum(raw_pred, 0.0) + 1e-6, 0.0)
        seeds.append(_seed_from(v))

    init_parts = list(seeds)
    for s in seeds:
        for _ in range(max(1, effective_pop // (4 * len(seeds)))):
            init_parts.append(
                _project_with_caps(
                    s + rng.normal(0, 0.02, size=n),
                    w_min=w_min,
                    w_max_vec=w_max_vec,
                )
            )
    init = np.vstack(init_parts)
    if len(init) < effective_pop:
        extra = rng.uniform(0, 1, size=(effective_pop - len(init), n)) * w_max_vec
        init = np.vstack(
            [init]
            + [
                _project_with_caps(row, w_min=w_min, w_max_vec=w_max_vec)
                for row in extra
            ]
        )
    init = init[:effective_pop]

    class SeededSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return init[:n_samples].copy()

    def _run(limit: float):
        algorithm = NSGA3(
            pop_size=effective_pop,
            ref_dirs=ref_dirs,
            sampling=SeededSampling(),
        )
        return minimize(
            PortfolioProblem(limit),
            algorithm,
            ("n_gen", int(cfg.n_gen)),
            seed=RANDOM_STATE,
            verbose=False,
        )

    res = _run(cvar_limit)
    pareto_x, pareto_f = res.X, res.F
    if pareto_x is None or len(np.atleast_2d(pareto_x)) == 0:
        logger.warning("Pareto vide sous CVaR — relance libre")
        res = _run(1e9)
        pareto_x, pareto_f = res.X, res.F

    if pareto_x is None or len(np.atleast_2d(pareto_x)) == 0:
        w_eq = _seed_from(np.ones(n))
        pareto_w = np.atleast_2d(w_eq)
        f_fallback = [
            -float(np.dot(w_eq, score)),
            portfolio_cvar(w_eq, ret_mat),
        ]
        if use_liq_obj:
            f_fallback.append(-float(np.dot(w_eq, L_sig)))
        pareto_f = np.array([f_fallback])
        n_pareto = 0
    else:
        pareto_x = np.atleast_2d(pareto_x)
        pareto_f = np.atleast_2d(pareto_f)
        pareto_w = np.vstack(
            [_project_with_caps(w, w_min=w_min, w_max_vec=w_max_vec) for w in pareto_x]
        )
        n_pareto = len(pareto_w)

    idx_agg = int(np.argmin(pareto_f[:, 0]))
    idx_def = int(np.argmin(pareto_f[:, 1]))
    idx_knee = _knee_point_index(pareto_f)
    idx_sel = _select_pareto_index(
        pareto_f,
        cfg.selection_rule,
        cvar_limit=cvar_limit,
        l_min=cfg.l_min,
    )
    idx_liq = (
        int(np.argmin(pareto_f[:, 2]))
        if use_liq_obj and pareto_f.shape[1] > 2
        else idx_knee
    )

    profiles = {
        "P_agressif": pareto_w[idx_agg],
        "P_equilibre": pareto_w[idx_knee],
        "P_defensif": pareto_w[idx_def],
        "P_selected": pareto_w[idx_sel],
        "P_max_liq": pareto_w[idx_liq],
    }

    def _pack(name: str, w: np.ndarray) -> dict[str, Any]:
        alpha_opt = float(np.dot(w, score))
        alpha_raw = float(np.dot(w, raw_pred))
        cvar = portfolio_cvar(w, ret_mat)
        to = 0.5 * float(np.abs(w - prev_w).sum()) if prev_w.sum() > 0 else 0.0
        L_port = float(np.dot(w, L_sig))
        held = [
            {
                "ticker": tickers[i],
                "weight": float(w[i]),
                "recommendation": recs[i],
                "predicted_return": float(raw_pred[i]),
                "conviction_score": float(conv[i]),
                "score_opt": float(score[i]),
                "liquidity_vmq_20j": float(liq[i]),
                "L_sigmoid": float(L_sig[i]),
            }
            for i in range(n)
            if w[i] >= cfg.weight_report_floor
        ]
        held.sort(key=lambda r: -r["weight"])
        return {
            "name": name,
            "n_positions": len(held),
            "sum_weights": float(w.sum()),
            "expected_alpha_opt": round(alpha_opt, 6),
            "expected_alpha_raw": round(alpha_raw, 6),
            "cvar_95": round(cvar, 6),
            "liquidity_L_weighted": round(L_port, 4),
            "turnover_vs_prev": round(to, 4),
            "cvar_ok": bool(cvar <= cvar_limit + 1e-9),
            "holdings": held,
            "weights_full": {tickers[i]: float(w[i]) for i in range(n)},
        }

    objectives = [
        "max preference-adjusted predicted_return",
        "min CVaR 95%",
    ]
    if use_liq_obj:
        objectives.append("max portfolio L=sigmoid(VMQ_20j)")

    liq_role = {
        "eligibility": "eligibility_filter_before_nsga",
        "pareto": "nsga_objective_max_L",
        "none": "ignored",
    }[cfg.liquidity_mode]

    return {
        "method": f"NSGA-III stage2 ({n_obj} obj; liquidity_mode={cfg.liquidity_mode})",
        "as_of": str(as_of.date()),
        "n_candidates": n,
        "n_pareto": n_pareto,
        "selection_rule": cfg.selection_rule,
        "sell_mode": cfg.sell_mode,
        "liquidity_mode": cfg.liquidity_mode,
        "cvar_masi": round(cvar_masi, 6),
        "cvar_limit": round(cvar_limit, 6),
        "cvar_limit_target": round(cvar_limit_target, 6),
        "constraint_relaxed": constraint_relaxed,
        "lookback_days": RETURNS_LOOKBACK_DAYS,
        "objectives": objectives,
        "constraints": {
            "sum_weights": 1.0,
            "w_min": w_min,
            "w_max": cfg.w_max,
            "w_max_SELL": cfg.w_max_SELL if cfg.sell_mode == "penalize" else None,
            "cvar_masi_ratio": CVAR_MASI_RATIO,
            "max_turnover": cfg.max_turnover,
            "min_vmq": cfg.min_vmq if cfg.liquidity_mode == "eligibility" else None,
            "l_min": cfg.l_min if cfg.liquidity_mode == "eligibility" else None,
            "liquidity_role": liq_role,
        },
        "preferences": {
            "BUY": cfg.pref_BUY,
            "NEUTRAL": cfg.pref_NEUTRAL,
            "SELL": cfg.pref_SELL,
        },
        "tickers": tickers,
        "recommendations": recs,
        "P_agressif": _pack("Agressif", profiles["P_agressif"]),
        "P_equilibre": _pack("Équilibré", profiles["P_equilibre"]),
        "P_defensif": _pack("Défensif", profiles["P_defensif"]),
        "P_selected": _pack("Selected", profiles["P_selected"]),
        "P_max_liq": _pack("MaxLiquidite", profiles["P_max_liq"]),
        "pareto_front": [
            {
                "sol_id": int(i),
                "is_knee": bool(i == idx_knee),
                "is_max_alpha": bool(i == idx_agg),
                "is_min_cvar": bool(i == idx_def),
                "is_max_liq": bool(i == idx_liq),
                "is_selected": bool(i == idx_sel),
                **{
                    k: v
                    for k, v in _pack(f"sol_{i:03d}", pareto_w[i]).items()
                    if k != "name"
                },
            }
            for i in range(len(pareto_w))
        ],
        "knee_algorithm": (
            "Distance euclidienne au point d'utopie après min-max "
            "sur chaque objectif du vecteur F (tous minimisés : "
            "-alpha, CVaR, -L). Aucun rendement réalisé futur."
        ),
    }


def _filter_month(
    stage2: pd.DataFrame,
    cell: int,
    date: pd.Timestamp,
) -> pd.DataFrame:
    return _filter_month_scorer(stage2, cell, date, id_col="cell")


def _filter_month_scorer(
    stage2: pd.DataFrame,
    scorer_id: Any,
    date: pd.Timestamp,
    *,
    id_col: str = "cell",
) -> pd.DataFrame:
    d = pd.Timestamp(date).normalize()
    s = stage2.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    return s[(s[id_col] == scorer_id) & (s["date"] == d)]


def run_stage2_scorer(
    stage2_inputs: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    scorer_id: Any,
    id_col: str = "cell",
    configuration: str | None = None,
    cfg: Stage2Config | None = None,
    reset_prev_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward mensuel pour un identifiant de scores (cellule C1–C4 ou model_id Bloc B).

    ``id_col`` : ``cell`` (legacy) ou ``model_id`` (Bloc D multi-modèles).
    """
    cfg = cfg or Stage2Config()
    panel = stage2_inputs[stage2_inputs[id_col] == scorer_id].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["mois"] = panel["date"].dt.to_period("M").astype(str)
    panel = panel[
        (panel["mois"] >= cfg.eval_start_month) & (panel["mois"] <= cfg.eval_end_month)
    ].copy()
    dates = sorted(panel["date"].dropna().unique())
    if not dates:
        raise ValueError(
            f"{id_col}={scorer_id}: aucune date dans "
            f"[{cfg.eval_start_month}, {cfg.eval_end_month}]"
        )

    if configuration is None:
        if id_col == "cell":
            configuration = CELL_META.get(int(scorer_id), {}).get("label", f"C{scorer_id}")
        else:
            configuration = str(scorer_id)

    groupe = None
    if id_col == "model_id" and "groupe" in panel.columns:
        gvals = panel["groupe"].dropna().unique()
        groupe = str(gvals[0]) if len(gvals) else None

    fold_id = panel["fold_id"].iloc[0] if "fold_id" in panel.columns else None

    weight_rows: list[dict[str, Any]] = []
    obj_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    pareto_obj_rows: list[dict[str, Any]] = []
    pareto_w_rows: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None

    opt_logger = logging.getLogger("bvc_recommender.models.portfolio_optimizer")
    prev_level = opt_logger.level
    opt_logger.setLevel(logging.WARNING)

    log_tag = f"{id_col}={scorer_id}"

    try:
        for i, dt in enumerate(dates, start=1):
            mois = pd.Timestamp(dt).strftime("%Y-%m")
            month = _filter_month_scorer(panel, scorer_id, dt, id_col=id_col)
            status = "ok"
            err = None
            try:
                uni = prepare_universe(month, cfg)
                result = optimize_universe(
                    uni, cours, indices, as_of=pd.Timestamp(dt), cfg=cfg, prev_weights=prev
                )
            except Exception as exc:  # noqa: BLE001
                status = "fallback_equal_buy_neutral"
                err = str(exc)
                logger.warning(
                    "[%s | %s] NSGA échec (%s) — equal-weight fallback",
                    log_tag,
                    mois,
                    exc,
                )
                uni = prepare_universe(month, cfg)
                if uni.empty:
                    meta.append(
                        {
                            id_col: scorer_id,
                            "cell": scorer_id if id_col == "cell" else pd.NA,
                            "model_id": scorer_id if id_col == "model_id" else pd.NA,
                            "groupe": groupe,
                            "fold_id": fold_id,
                            "mois": mois,
                            "date_rebalance": str(pd.Timestamp(dt).date()),
                            "status": "empty_universe",
                            "error": err,
                        }
                    )
                    continue
                n = len(uni)
                w = 1.0 / n
                result = {
                    "n_candidates": n,
                    "n_pareto": 0,
                    "selection_rule": cfg.selection_rule,
                    "sell_mode": cfg.sell_mode,
                    "P_selected": {
                        "name": "Selected",
                        "n_positions": n,
                        "sum_weights": 1.0,
                        "expected_alpha_opt": float(np.nansum(uni["score_opt"] * w)),
                        "expected_alpha_raw": float(np.nansum(uni["predicted_return"] * w)),
                        "cvar_95": None,
                        "liquidity_L_weighted": None,
                        "turnover_vs_prev": None,
                        "holdings": [
                            {
                                "ticker": str(r.ticker),
                                "weight": w,
                                "recommendation": str(r.recommendation),
                                "predicted_return": float(r.predicted_return),
                                "conviction_score": float(r.conviction_score),
                                "score_opt": float(r.score_opt),
                                "liquidity_vmq_20j": float(r.liquidity_vmq_20j)
                                if pd.notna(r.liquidity_vmq_20j)
                                else np.nan,
                                "L_sigmoid": float(r.L_sigmoid)
                                if hasattr(r, "L_sigmoid") and pd.notna(r.L_sigmoid)
                                else np.nan,
                            }
                            for r in uni.itertuples()
                        ],
                        "weights_full": {str(t): w for t in uni["ticker"]},
                    },
                    "P_agressif": None,
                    "P_equilibre": None,
                    "P_defensif": None,
                    "P_max_liq": None,
                    "pareto_front": [],
                }
                for k in ("P_agressif", "P_equilibre", "P_defensif", "P_max_liq"):
                    result[k] = dict(result["P_selected"])
                    result[k]["name"] = k

            for port_key in (
                "P_agressif",
                "P_equilibre",
                "P_defensif",
                "P_selected",
                "P_max_liq",
            ):
                p = result.get(port_key) or {}
                for h in p.get("holdings") or []:
                    row = {
                        id_col: scorer_id,
                        "cell": int(scorer_id) if id_col == "cell" else pd.NA,
                        "model_id": scorer_id if id_col == "model_id" else pd.NA,
                        "groupe": groupe,
                        "fold_id": fold_id,
                        "configuration": configuration,
                        "mois": mois,
                        "date_rebalance": str(pd.Timestamp(dt).date()),
                        "portfolio": port_key,
                        "ticker": h["ticker"],
                        "weight": h["weight"],
                        "recommendation": h["recommendation"],
                        "predicted_return": h["predicted_return"],
                        "conviction_score": h.get("conviction_score"),
                        "score_opt": h.get("score_opt"),
                        "liquidity_vmq_20j": h.get("liquidity_vmq_20j"),
                        "L_sigmoid": h.get("L_sigmoid"),
                    }
                    weight_rows.append(row)
                obj_rows.append(
                    {
                        id_col: scorer_id,
                        "cell": int(scorer_id) if id_col == "cell" else pd.NA,
                        "model_id": scorer_id if id_col == "model_id" else pd.NA,
                        "groupe": groupe,
                        "fold_id": fold_id,
                        "configuration": configuration,
                        "mois": mois,
                        "date_rebalance": str(pd.Timestamp(dt).date()),
                        "portfolio": port_key,
                        "n_candidates": result.get("n_candidates"),
                        "n_positions": p.get("n_positions"),
                        "sum_weights": p.get("sum_weights"),
                        "expected_alpha_opt": p.get("expected_alpha_opt"),
                        "expected_alpha_raw": p.get("expected_alpha_raw"),
                        "cvar_95": p.get("cvar_95"),
                        "liquidity_L_weighted": p.get("liquidity_L_weighted"),
                        "turnover_vs_prev": p.get("turnover_vs_prev"),
                        "n_pareto": result.get("n_pareto"),
                        "selection_rule": result.get("selection_rule"),
                        "sell_mode": result.get("sell_mode"),
                        "status": status,
                        "error": err,
                    }
                )

            for sol in result.get("pareto_front") or []:
                pareto_obj_rows.append(
                    {
                        id_col: scorer_id,
                        "cell": int(scorer_id) if id_col == "cell" else pd.NA,
                        "model_id": scorer_id if id_col == "model_id" else pd.NA,
                        "fold_id": fold_id,
                        "mois": mois,
                        "date_rebalance": str(pd.Timestamp(dt).date()),
                        "sol_id": sol.get("sol_id"),
                        "is_knee": sol.get("is_knee"),
                        "is_max_alpha": sol.get("is_max_alpha"),
                        "is_min_cvar": sol.get("is_min_cvar"),
                        "is_max_liq": sol.get("is_max_liq"),
                        "is_selected": sol.get("is_selected"),
                        "n_positions": sol.get("n_positions"),
                        "sum_weights": sol.get("sum_weights"),
                        "expected_alpha_opt": sol.get("expected_alpha_opt"),
                        "expected_alpha_raw": sol.get("expected_alpha_raw"),
                        "cvar_95": sol.get("cvar_95"),
                        "liquidity_L_weighted": sol.get("liquidity_L_weighted"),
                        "turnover_vs_prev": sol.get("turnover_vs_prev"),
                    }
                )
                for h in sol.get("holdings") or []:
                    pareto_w_rows.append(
                        {
                            id_col: scorer_id,
                            "fold_id": fold_id,
                            "mois": mois,
                            "date_rebalance": str(pd.Timestamp(dt).date()),
                            "sol_id": sol.get("sol_id"),
                            "is_selected": sol.get("is_selected"),
                            "is_knee": sol.get("is_knee"),
                            "ticker": h["ticker"],
                            "weight": h["weight"],
                            "recommendation": h.get("recommendation"),
                            "score_opt": h.get("score_opt"),
                            "liquidity_vmq_20j": h.get("liquidity_vmq_20j"),
                            "L_sigmoid": h.get("L_sigmoid"),
                        }
                    )

            sel = result.get("P_selected") or {}
            if reset_prev_weights or i == 1:
                pass  # prev unchanged except after first month in block
            prev = sel.get("weights_full") or {
                h["ticker"]: h["weight"] for h in (sel.get("holdings") or [])
            }
            meta.append(
                {
                    id_col: scorer_id,
                    "cell": int(scorer_id) if id_col == "cell" else pd.NA,
                    "model_id": scorer_id if id_col == "model_id" else pd.NA,
                    "groupe": groupe,
                    "fold_id": fold_id,
                    "mois": mois,
                    "date_rebalance": str(pd.Timestamp(dt).date()),
                    "status": status,
                    "error": err,
                    "n_candidates": result.get("n_candidates"),
                    "n_positions_selected": sel.get("n_positions"),
                    "expected_alpha_selected": sel.get("expected_alpha_raw"),
                    "cvar_selected": sel.get("cvar_95"),
                    "liquidity_L_selected": sel.get("liquidity_L_weighted"),
                    "n_pareto": result.get("n_pareto"),
                    "turnover_vs_prev": sel.get("turnover_vs_prev"),
                }
            )
            logger.info(
                "[BlocD NSGA] %s fold=%s mois=%s status=%s n_pareto=%s pos=%s",
                log_tag,
                fold_id,
                mois,
                status,
                result.get("n_pareto"),
                sel.get("n_positions"),
            )
    finally:
        opt_logger.setLevel(prev_level)

    return (
        pd.DataFrame(weight_rows),
        pd.DataFrame(obj_rows),
        meta,
        pd.DataFrame(pareto_obj_rows),
        pd.DataFrame(pareto_w_rows),
    )


def run_stage2_cell(
    stage2_inputs: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    cell_id: int,
    cfg: Stage2Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Walk-forward mensuel pour une cellule C1–C4 → weights, objectives, meta."""
    return run_stage2_scorer(
        stage2_inputs,
        cours,
        indices,
        scorer_id=cell_id,
        id_col="cell",
        cfg=cfg,
        reset_prev_weights=False,
    )


def run_stage2_model(
    stage2_inputs: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    model_id: str,
    cfg: Stage2Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Walk-forward mensuel pour un ``model_id`` Bloc B (bloc de test isolé)."""
    return run_stage2_scorer(
        stage2_inputs,
        cours,
        indices,
        scorer_id=model_id,
        id_col="model_id",
        configuration=model_id,
        cfg=cfg,
        reset_prev_weights=True,
    )

def run_stage2_all_cells(
    stage2_inputs: pd.DataFrame,
    *,
    cell_ids: tuple[int, ...] = (1, 2, 3, 4),
    cfg: Stage2Config | None = None,
) -> dict[str, Any]:
    cfg = cfg or Stage2Config()
    cours, indices, _technical = load_market_panels()
    all_w, all_o, all_m, all_po, all_pw = [], [], [], [], []
    for cell_id in cell_ids:
        logger.info("=== Stage2 NSGA cellule %s (%s) ===", cell_id, CELL_META.get(cell_id, {}).get("label"))
        w, o, m, po, pw = run_stage2_cell(
            stage2_inputs, cours, indices, cell_id=cell_id, cfg=cfg
        )
        all_w.append(w)
        all_o.append(o)
        all_m.extend(m)
        all_po.append(po)
        all_pw.append(pw)
    return {
        "weights": pd.concat(all_w, ignore_index=True) if all_w else pd.DataFrame(),
        "objectives": pd.concat(all_o, ignore_index=True) if all_o else pd.DataFrame(),
        "month_meta": pd.DataFrame(all_m),
        "pareto_objectives": pd.concat(all_po, ignore_index=True) if all_po else pd.DataFrame(),
        "pareto_weights": pd.concat(all_pw, ignore_index=True) if all_pw else pd.DataFrame(),
        "config": asdict(cfg),
    }


def save_stage2_artifacts(
    results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for cell in sorted(results["weights"]["cell"].dropna().unique()):
        cell = int(cell)
        label = f"C{cell}"
        w = results["weights"][results["weights"]["cell"] == cell]
        o = results["objectives"][results["objectives"]["cell"] == cell]
        wp = output_dir / f"stage2_weights_{label}.parquet"
        op = output_dir / f"stage2_objectives_{label}.parquet"
        w.to_parquet(wp, index=False)
        o.to_parquet(op, index=False)
        # CSV lisible P_selected
        sel = w[w["portfolio"] == "P_selected"]
        cp = output_dir / f"stage2_P_selected_{label}.csv"
        sel.to_csv(cp, index=False)
        paths[f"weights_{label}"] = wp
        paths[f"objectives_{label}"] = op
        paths[f"selected_csv_{label}"] = cp

    meta_path = output_dir / "stage2_month_meta.csv"
    results["month_meta"].to_csv(meta_path, index=False)
    paths["month_meta"] = meta_path

    # All-in-one
    all_w = output_dir / "stage2_weights_all.parquet"
    all_o = output_dir / "stage2_objectives_all.parquet"
    results["weights"].to_parquet(all_w, index=False)
    results["objectives"].to_parquet(all_o, index=False)
    paths["weights_all"] = all_w
    paths["objectives_all"] = all_o

    po = results.get("pareto_objectives")
    pw = results.get("pareto_weights")
    if po is not None and len(po):
        pop = output_dir / "stage2_pareto_objectives.parquet"
        po.to_parquet(pop, index=False)
        paths["pareto_objectives"] = pop
    if pw is not None and len(pw):
        pwp = output_dir / "stage2_pareto_weights.parquet"
        pw.to_parquet(pwp, index=False)
        paths["pareto_weights"] = pwp

    mode = results["config"].get("liquidity_mode", "eligibility")
    if mode == "eligibility":
        liq_proto = "eligibility constraint VMQ_20j >= min_vmq (not NSGA objective)"
        nsga_objs = ["max preference-adjusted predicted_return", "min CVaR 95%"]
    elif mode == "pareto":
        liq_proto = "NSGA objective max L=sigmoid(VMQ); no VMQ eligibility filter"
        nsga_objs = [
            "max preference-adjusted predicted_return",
            "min CVaR 95%",
            "max portfolio L=sigmoid(VMQ_20j)",
        ]
    else:
        liq_proto = "ignored (no filter, not an NSGA objective)"
        nsga_objs = ["max preference-adjusted predicted_return", "min CVaR 95%"]

    summary = {
        "protocol": {
            "eval_window": f"{results['config'].get('eval_start_month')} → {results['config'].get('eval_end_month')}",
            "liquidity_mode": mode,
            "liquidity": liq_proto,
            "min_vmq_mad": results["config"].get("min_vmq") if mode == "eligibility" else None,
            "l_min": results["config"].get("l_min") if mode == "eligibility" else None,
            "sell": "excluded before NSGA when sell_mode=exclude",
            "nsga_objectives": nsga_objs,
            "no_vmq_imputation": True,
            "window_frozen_before_backtest": True,
        },
        "config": results["config"],
        "n_weight_rows": int(len(results["weights"])),
        "n_months_total": int(results["month_meta"]["mois"].nunique())
        if len(results["month_meta"])
        else 0,
        "cells": {},
        "lookahead": {
            "returns_history": "cours/indices filtrés date <= as_of",
            "vmq": "liquidity_vmq_20j as-of t (stage2_inputs); no future",
            "scores": "predicted_return / recommendation de l'étage 1 à t",
            "no_future_returns_in_opt": True,
            "no_vmq_imputation": True,
        },
    }
    for cell, grp in results["month_meta"].groupby("cell"):
        summary["cells"][str(int(cell))] = {
            "label": CELL_META.get(int(cell), {}).get("label"),
            "n_months": int(len(grp)),
            "n_ok": int((grp["status"] == "ok").sum()),
            "n_fallback": int((grp["status"] != "ok").sum()),
            "mean_positions_selected": float(
                pd.to_numeric(grp["n_positions_selected"], errors="coerce").mean()
            ),
        }
    sp = output_dir / "stage2_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary"] = sp
    return paths
