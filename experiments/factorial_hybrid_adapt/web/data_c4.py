"""
Chargeurs lecture seule — recommandations meta (meilleur algo par titre) + portefeuille champion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.factorial_hybrid_adapt import bloc_b_models

MODEL_IDS: tuple[str, ...] = bloc_b_models.MODEL_IDS
STRATEGY_LABELS: dict[str, str] = bloc_b_models.STRATEGY_LABELS

BASE = Path(__file__).resolve().parents[1]
MART = BASE / "outputs" / "platform_mart"
MART_BEST = MART / "best"
MART_C1C5 = MART / "c1c5"
REPORTS = BASE / "outputs" / "reports"
BLOC_D_PARTS = BASE / "outputs" / "bloc_d_wf_14blocks"
CACHE_VER = "best_algo_mart_v3_masi_ext"

BLOC_D_MODEL_IDS: tuple[str, ...] = MODEL_IDS
BLOC_D_LABELS: dict[str, str] = dict(STRATEGY_LABELS)
BLOC_D_COLORS: dict[str, str] = {
    "c1_ridge": "#1f77b4",
    "c2_rf": "#b45309",
    "c3_lightgbm": "#ff7f0e",
    "c4_hybrid_tri": "#15803d",
    "c5_hybrid_tri_regime": "#7c3aed",
    "MASI": "#1d4ed8",
}
PRINCIPAL_BLOC_D_MODEL = "c3_lightgbm"


def _best_ready() -> bool:
    return (MART_BEST / "meta_platform.json").is_file()


def _titles_path() -> Path:
    if _best_ready():
        return MART_BEST / "fact_titles_month.parquet"
    if (MART_C1C5 / "fact_titles_month.parquet").is_file():
        return MART_C1C5 / "fact_titles_month.parquet"
    return MART / "fact_titles_month_C4.parquet"


def _holdings_path() -> Path:
    if _best_ready():
        return MART_BEST / "fact_portfolio_holdings.parquet"
    if (MART_C1C5 / "fact_portfolio_holdings.parquet").is_file():
        return MART_C1C5 / "fact_portfolio_holdings.parquet"
    return MART / "fact_portfolio_holdings_C4.parquet"


def _monthly_path() -> Path:
    if _best_ready():
        return MART_BEST / "fact_portfolio_monthly.parquet"
    if (MART_C1C5 / "fact_portfolio_monthly.parquet").is_file():
        return MART_C1C5 / "fact_portfolio_monthly.parquet"
    return MART / "fact_portfolio_monthly_C4.parquet"


@st.cache_data(show_spinner=False)
def load_meta(_ver: str = CACHE_VER) -> dict:
    if _best_ready():
        meta = json.loads((MART_BEST / "meta_platform.json").read_text(encoding="utf-8"))
    elif (MART_C1C5 / "meta_c1c5.json").is_file():
        meta = json.loads((MART_C1C5 / "meta_c1c5.json").read_text(encoding="utf-8"))
    else:
        p = MART / "meta_platform.json"
        meta = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    meta["_cache_ver"] = _ver
    return meta


@st.cache_data(show_spinner=False)
def list_months(_ver: str = CACHE_VER) -> list[str]:
    meta = load_meta(_ver)
    months = meta.get("months")
    if months:
        return list(months)
    p = _monthly_path()
    if p.is_file():
        return sorted(pd.read_parquet(p, columns=["month"])["month"].astype(str).unique())
    return []


@st.cache_data(show_spinner=False)
def load_titles(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = _titles_path()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "model_id" in df.columns and "selected_model_id" not in df.columns:
        if df["model_id"].nunique() > 1:
            df = _best_per_title_from_c1c5(df)
        else:
            df = df.copy()
    return df


def _best_per_title_from_c1c5(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback si mart best/ non généré : argmax alpha sur C1–C5."""
    work = df.copy()
    work["alpha"] = pd.to_numeric(work["alpha"], errors="coerce")
    idx = work.groupby(["month", "ticker"], sort=False)["alpha"].idxmax()
    best = work.loc[idx].copy()
    best["selected_model_id"] = best.get("model_id")
    best["selected_strategy"] = best.get("strategy_label")
    return best


@st.cache_data(show_spinner=False)
def load_holdings(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = _holdings_path()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "model_id" in df.columns and df["model_id"].nunique() > 1:
        champion = load_meta(_ver).get("portfolio_champion_model_id", PRINCIPAL_BLOC_D_MODEL)
        df = df[df["model_id"] == champion].copy()
    return df


@st.cache_data(show_spinner=False)
def load_monthly(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = _monthly_path()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "model_id" in df.columns and df["model_id"].nunique() > 1:
        champion = load_meta(_ver).get("portfolio_champion_model_id", PRINCIPAL_BLOC_D_MODEL)
        df = df[df["model_id"] == champion].copy()
    return df


@st.cache_data(show_spinner=False)
def load_recommendations(_ver: str = CACHE_VER) -> pd.DataFrame:
    if _best_ready() and (MART_BEST / "fact_recommendations.parquet").is_file():
        return pd.read_parquet(MART_BEST / "fact_recommendations.parquet")
    return load_titles(_ver)


@st.cache_data(show_spinner=False)
def load_daily_wealth(_ver: str = CACHE_VER) -> pd.DataFrame:
    wealth = load_bloc_d_daily_wealth(_ver)
    if not wealth.empty:
        return wealth
    p = MART / "fact_performance_daily.parquet"
    if p.is_file():
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_benchmarks(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = MART_BEST / "kpi_champion.csv"
    if p.is_file():
        return pd.read_csv(p)
    p2 = MART_C1C5 / "kpi_c1c5.csv"
    if p2.is_file():
        return pd.read_csv(p2)
    return pd.read_csv(MART / "kpi_benchmarks.csv") if (MART / "kpi_benchmarks.csv").is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_kpi_cells(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = MART_C1C5 / "kpi_c1c5.csv"
    if p.is_file():
        return pd.read_csv(p)
    p = MART / "kpi_C1C4.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tests(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = MART / "tests_statistiques.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_subperiods(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = MART / "subperiods_C1C4.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_liquidity_robustness(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = MART / "robustness_liquidity.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_masi_long(_ver: str = CACHE_VER) -> pd.DataFrame:
    for p in (MART_BEST / "fact_masi_since_2010.csv", MART / "fact_masi_since_2010.csv"):
        if p.is_file():
            return pd.read_csv(p, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_price_panel(_ver: str = CACHE_VER) -> pd.DataFrame:
    from experiments.factorial_hybrid_adapt.allocation import load_market_panels
    from experiments.factorial_hybrid_adapt.stage2_metrics import _price_panel

    cours, _indices, _tech = load_market_panels()
    return _price_panel(cours)


@st.cache_data(show_spinner=False)
def load_bloc_d_summary(_ver: str = CACHE_VER) -> pd.DataFrame:
    p = REPORTS / "bloc_d_summary_14blocks.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    order = {m: i for i, m in enumerate(BLOC_D_MODEL_IDS)}
    df["_ord"] = df["model_id"].map(lambda x: order.get(str(x), 99))
    return df.sort_values("_ord").drop(columns=["_ord"])


@st.cache_data(show_spinner=False)
def load_bloc_d_benchmarks(_ver: str = CACHE_VER) -> dict:
    """Moyennes MASI / EW MASI20 sur les 14 blocs (rapport Bloc D)."""
    p = REPORTS / "bloc_d_report_14blocks.json"
    if not p.is_file():
        return {}
    rep = json.loads(p.read_text(encoding="utf-8"))
    return dict(rep.get("benchmark_summary") or {})


@st.cache_data(show_spinner=False)
def load_bloc_d_sharpe_chart(_ver: str = CACHE_VER) -> pd.DataFrame:
    """Sharpe moyen 14 blocs — C1–C5 + MASI (ordre d'affichage fixe)."""
    summary = load_bloc_d_summary(_ver)
    bench = load_bloc_d_benchmarks(_ver)
    rows: list[dict] = []
    for mid in BLOC_D_MODEL_IDS:
        sub = summary.loc[summary["model_id"] == mid]
        if sub.empty:
            continue
        rows.append(
            {
                "Strategie": BLOC_D_LABELS.get(mid, mid),
                "model_id": mid,
                "Sharpe": float(sub.iloc[0]["sharpe"]),
                "color": BLOC_D_COLORS.get(mid, "#64748b"),
            }
        )
    masi_sharpe = bench.get("masi_sharpe_mean")
    if masi_sharpe is not None:
        rows.append(
            {
                "Strategie": "MASI",
                "model_id": "MASI",
                "Sharpe": float(masi_sharpe),
                "color": BLOC_D_COLORS["MASI"],
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner="Construction courbes C1–C5…")
def load_bloc_d_daily_wealth(_ver: str = CACHE_VER) -> pd.DataFrame:
    if not BLOC_D_PARTS.is_dir():
        return pd.DataFrame()

    from experiments.factorial_hybrid_adapt.allocation import load_market_panels
    from experiments.factorial_hybrid_adapt.bloc_d_multi_model import (
        simulate_benchmark_block,
        simulate_block_portfolio,
    )
    from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset
    from experiments.factorial_hybrid_adapt.stage2_bloc_b_loader import block_fold_ids_from_summary
    from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel
    from experiments.factorial_hybrid_adapt.walk_forward_folds import (
        available_months_from_dates,
        generate_rolling_folds,
    )

    block_ids = block_fold_ids_from_summary()
    months = available_months_from_dates(load_ml_dataset()["date_cours"])
    by_id = {f.fold_id: f for f in generate_rolling_folds(months)}
    blocks = [by_id[fid] for fid in block_ids if fid in by_id]

    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    series: dict[str, pd.Series] = {}
    for model_id in BLOC_D_MODEL_IDS:
        daily_parts: list[pd.Series] = []
        for fold in blocks:
            wpath = BLOC_D_PARTS / f"fold_{fold.fold_id:04d}_{model_id}_weights.parquet"
            if not wpath.is_file():
                continue
            weights = pd.read_parquet(wpath)
            sim = simulate_block_portfolio(
                weights, prices, technical, masi,
                model_id=model_id, fold_id=fold.fold_id,
            )
            dr = sim.get("daily_returns")
            if dr is not None and not dr.empty:
                daily_parts.append(dr)
        if not daily_parts:
            continue
        daily = pd.concat(daily_parts).sort_index()
        daily = daily[~daily.index.duplicated(keep="last")]
        label = BLOC_D_LABELS.get(model_id, model_id)
        series[label] = 100 * (1 + daily).cumprod()

    masi_parts: list[pd.Series] = []
    for fold in blocks:
        bsim = simulate_benchmark_block(fold, prices, cours, masi, kind="masi")
        dr = bsim.get("daily_returns")
        if dr is not None and not dr.empty:
            masi_parts.append(dr)
    if masi_parts:
        masi_daily = pd.concat(masi_parts).sort_index()
        masi_daily = masi_daily[~masi_daily.index.duplicated(keep="last")]
        # Les stratégies prolongent le dernier mois jusqu'au max des cours ;
        # aligner le MASI sur le même horizon (données officielles + extension).
        if series:
            horizon = max(s.index.max() for s in series.values())
            last_m = masi_daily.index.max()
            extra = masi.loc[(masi.index > last_m) & (masi.index <= horizon)]
            if not extra.empty:
                masi_daily = pd.concat([masi_daily, extra]).sort_index()
                masi_daily = masi_daily[~masi_daily.index.duplicated(keep="last")]
        series["MASI"] = 100 * (1 + masi_daily).cumprod()

    if not series:
        return pd.DataFrame()

    wealth = pd.DataFrame(series)
    wealth.index = pd.to_datetime(wealth.index)
    wealth = wealth.sort_index()
    for col in wealth.columns:
        wealth[f"dd_{col}"] = wealth[col] / wealth[col].cummax() - 1
    wealth = wealth.reset_index()
    if wealth.columns[0] != "date":
        wealth = wealth.rename(columns={wealth.columns[0]: "date"})
    return wealth


@st.cache_data(show_spinner=False)
def load_bloc_d_kpi_table(_ver: str = CACHE_VER) -> pd.DataFrame:
    summary = load_bloc_d_summary(_ver)
    wealth = load_bloc_d_daily_wealth(_ver)
    if summary.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in summary.iterrows():
        mid = str(r["model_id"])
        label = BLOC_D_LABELS.get(mid, mid)
        w_final = None
        if not wealth.empty and label in wealth.columns:
            w_final = float(wealth[label].dropna().iloc[-1])
        rows.append(
            {
                "Strategie": label,
                "model_id": mid,
                "groupe": r.get("groupe"),
                "Sharpe": float(r["sharpe"]),
                "Rendement": float(r["ann_return"]),
                "Sortino": float(r.get("sortino", 0)),
                "Max_DD": float(r["max_drawdown"]),
                "CVaR": float(r["cvar_95_realized"]),
                "Turnover": float(r["turnover_mean"]),
                "Liquidite": float(r["liquidity_mean"]),
                "wealth_final_100": w_final,
                "n_blocks": int(r.get("n_blocks", 14)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_decision_calendar(_ver: str = CACHE_VER) -> pd.DataFrame:
    t = load_titles(_ver)
    if t.empty:
        return pd.DataFrame(columns=["month", "decision_date"])
    date_col = "date" if "date" in t.columns else None
    if not date_col:
        return pd.DataFrame(columns=["month", "decision_date"])
    cal = (
        t[["month", date_col]]
        .drop_duplicates("month")
        .assign(month=lambda d: d["month"].astype(str))
        .assign(decision_date=lambda d: pd.to_datetime(d[date_col]))
        .sort_values("decision_date")
    )
    return cal[["month", "decision_date"]]


@st.cache_data(show_spinner=False)
def load_titles_month(model_id: str | None = None, _ver: str = CACHE_VER) -> pd.DataFrame:
    """Rétrocompat — meta sélection : model_id ignoré si mart best/."""
    df = load_titles(_ver)
    if model_id and "selected_model_id" in df.columns:
        return df[df["selected_model_id"] == model_id].copy()
    if model_id and "model_id" in df.columns:
        return df[df["model_id"] == model_id].copy()
    return df
