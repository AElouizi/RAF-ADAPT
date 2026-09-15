"""
Moteur de simulation buy-and-hold pour l'onglet Simulation.

Hypothèses V1 :
- recommandations figées au dernier mois de décision ≤ date d'entrée ;
- prix d'entrée / sortie = dernier cours disponible ≤ date ;
- actions entières + cash résiduel ;
- frais proportionnels au montant investi en actions (entrée ; sortie optionnelle) ;
- pas d'utilisation de realized_alpha pour la sélection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST

REC_LABEL_FR = {"BUY": "Achat", "NEUTRAL": "Neutre", "SELL": "Vente"}
REC_EMOJI = {"BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴"}
MIN_SIM_START = pd.Timestamp("2018-07-31")
DEFAULT_FEE = float(TRANSACTION_COST)  # 0.3 %


@dataclass
class PositionDetail:
    ticker: str
    name: str
    recommendation: str
    weight_target: float
    price_entry: float
    price_exit: float
    quantity: int
    amount_invested: float
    fee_entry: float
    value_exit: float
    fee_exit: float
    pnl: float
    contribution: float  # part du PnL total (hors cash)


@dataclass
class PortfolioResult:
    name: str
    capital: float
    cash: float
    fee_total: float
    value_final: float
    pnl: float
    total_return: float
    wealth: pd.Series
    positions: list[PositionDetail] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def label_reco_fr(rec: str) -> str:
    r = str(rec).upper()
    emoji = REC_EMOJI.get(r, "")
    fr = REC_LABEL_FR.get(r, r)
    return f"{emoji} {fr}".strip()


def resolve_as_of(
    start: pd.Timestamp,
    calendar: pd.DataFrame,
) -> tuple[str, pd.Timestamp, str]:
    """
    Retourne (month_key, entry_date_effective, note).

    Anti look-ahead : on n'utilise que les décisions dont la date ≤ start.
    Si start précède la première décision, on ancre l'entrée sur cette date.
    """
    start = pd.Timestamp(start).normalize()
    if calendar.empty:
        raise ValueError("Calendrier de décisions vide.")

    cal = calendar.copy()
    cal["decision_date"] = pd.to_datetime(cal["decision_date"]).dt.normalize()
    eligible = cal[cal["decision_date"] <= start]
    if not eligible.empty:
        row = eligible.iloc[-1]
        return str(row["month"]), start, (
            f"Recommandations du mois {row['month']} "
            f"(décision du {row['decision_date'].date()}, "
            f"disponible à la date de début)."
        )

    row = cal.iloc[0]
    entry = pd.Timestamp(row["decision_date"]).normalize()
    return str(row["month"]), entry, (
        f"Date de début antérieure à la première décision : "
        f"entrée effective le {entry.date()} (mois {row['month']})."
    )


def equal_weights(tickers: list[str]) -> dict[str, float]:
    tickers = [str(t) for t in tickers if t]
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def normalize_weights(weights: dict[str, float], *, tol: float = 1e-6) -> dict[str, float]:
    clean = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    s = sum(clean.values())
    if s <= 0:
        return {}
    out = {k: v / s for k, v in clean.items()}
    if abs(sum(out.values()) - 1.0) > tol:
        # correction numérique sur le dernier
        keys = list(out)
        out[keys[-1]] += 1.0 - sum(out.values())
    return out


def validate_weights_sum(weights: dict[str, float], *, tol: float = 1e-3) -> tuple[bool, float]:
    s = float(sum(float(v) for v in weights.values()))
    return abs(s - 1.0) <= tol, s


def price_asof(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> float | None:
    if ticker not in prices.columns:
        return None
    ser = prices[ticker].dropna()
    ser = ser[ser.index <= as_of]
    if ser.empty:
        return None
    val = float(ser.iloc[-1])
    return val if val > 0 and np.isfinite(val) else None


def build_buy_hold(
    *,
    name: str,
    capital: float,
    weights: dict[str, float],
    prices: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    fee_rate: float = DEFAULT_FEE,
    fee_on_exit: bool = False,
    names: dict[str, str] | None = None,
    recommendations: dict[str, str] | None = None,
    masi_returns: pd.Series | None = None,
) -> PortfolioResult:
    """Simule un portefeuille buy-and-hold avec actions entières."""
    entry_date = pd.Timestamp(entry_date).normalize()
    exit_date = pd.Timestamp(exit_date).normalize()
    names = names or {}
    recommendations = recommendations or {}
    weights = normalize_weights(weights)

    notes: list[str] = [
        "Buy-and-hold : pas de rebalancement entre l'entrée et la sortie.",
        "Quantités = parties entières ; le reliquat reste en cash (0 % de rendement).",
        f"Frais d'entrée = {100 * fee_rate:.2f} % du montant investi en actions.",
    ]
    if fee_on_exit:
        notes.append(f"Frais de sortie = {100 * fee_rate:.2f} % de la valeur liquidée.")
    else:
        notes.append("Aucun frais de sortie appliqué (V1).")

    skipped: list[str] = []
    positions: list[PositionDetail] = []
    invested_total = 0.0
    fee_entry_total = 0.0
    holdings_qty: dict[str, int] = {}

    for ticker, w in weights.items():
        px_in = price_asof(prices, ticker, entry_date)
        if px_in is None:
            skipped.append(f"{ticker} : pas de prix ≤ {entry_date.date()}")
            continue
        budget = capital * w
        # qty telle que qty*price*(1+fee) ≤ budget
        unit_cost = px_in * (1.0 + fee_rate)
        qty = int(np.floor(budget / unit_cost)) if unit_cost > 0 else 0
        if qty <= 0:
            skipped.append(f"{ticker} : budget insuffisant pour 1 action @ {px_in:.2f}")
            continue
        amount = qty * px_in
        fee_e = amount * fee_rate
        invested_total += amount
        fee_entry_total += fee_e
        holdings_qty[ticker] = qty

        px_out = price_asof(prices, ticker, exit_date)
        if px_out is None:
            px_out = px_in
            skipped.append(f"{ticker} : prix de sortie manquant, dernier prix d'entrée repris")
        value_out = qty * px_out
        fee_x = value_out * fee_rate if fee_on_exit else 0.0
        pnl = (value_out - fee_x) - (amount + fee_e)
        positions.append(
            PositionDetail(
                ticker=ticker,
                name=str(names.get(ticker, ticker)),
                recommendation=str(recommendations.get(ticker, "")),
                weight_target=w,
                price_entry=px_in,
                price_exit=px_out,
                quantity=qty,
                amount_invested=amount,
                fee_entry=fee_e,
                value_exit=value_out,
                fee_exit=fee_x,
                pnl=pnl,
                contribution=0.0,
            )
        )

    cash = capital - invested_total - fee_entry_total
    if cash < -1e-6:
        cash = 0.0

    wealth = _wealth_path(holdings_qty, cash, prices, entry_date, exit_date)
    if wealth.empty:
        value_final = cash
    else:
        value_final = float(wealth.iloc[-1])

    fee_exit_total = sum(p.fee_exit for p in positions)
    if fee_on_exit and fee_exit_total > 0 and not wealth.empty:
        # la valeur finale affichée est nette de frais de sortie (appliqués à la liquidation)
        value_final = float(wealth.iloc[-1]) - fee_exit_total

    fee_total = fee_entry_total + fee_exit_total
    pnl = value_final - capital
    total_return = (value_final / capital - 1.0) if capital > 0 else 0.0

    pnl_sum = sum(p.pnl for p in positions)
    for p in positions:
        p.contribution = (p.pnl / pnl_sum) if abs(pnl_sum) > 1e-12 else 0.0

    metrics: dict[str, Any] = {}
    if len(wealth) >= 2:
        rets = wealth.pct_change().dropna()
        bench = None
        if masi_returns is not None and not masi_returns.empty:
            bench = masi_returns.reindex(rets.index).fillna(0.0)
        else:
            bench = pd.Series(0.0, index=rets.index)
        metrics = compute_backtest_metrics(rets, bench, name=name)
        metrics["total_return"] = round(total_return, 4)
        metrics["volatility_ann"] = metrics.get("volatility_ann")
        metrics["sharpe"] = metrics.get("sharpe")
        metrics["max_drawdown"] = metrics.get("max_drawdown")
        metrics["annualized_return"] = metrics.get("annualized_return")

    return PortfolioResult(
        name=name,
        capital=capital,
        cash=cash,
        fee_total=fee_total,
        value_final=value_final,
        pnl=pnl,
        total_return=total_return,
        wealth=wealth,
        positions=positions,
        metrics=metrics,
        notes=notes,
        skipped=skipped,
    )


def _wealth_path(
    qty: dict[str, int],
    cash: float,
    prices: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> pd.Series:
    if not qty:
        idx = prices.index[(prices.index >= entry_date) & (prices.index <= exit_date)]
        if len(idx) == 0:
            return pd.Series(dtype=float)
        return pd.Series(cash, index=idx, dtype=float)

    tickers = [t for t in qty if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    sub = prices.loc[(prices.index >= entry_date) & (prices.index <= exit_date), tickers].copy()
    if sub.empty:
        # ancrage sur le premier/dernier point disponible
        sub = prices.loc[(prices.index <= exit_date), tickers].tail(1)
    sub = sub.ffill().bfill()
    values = sum(qty[t] * sub[t] for t in tickers) + cash
    return values.astype(float)


def masi_buy_hold_wealth(
    masi_long: pd.DataFrame,
    *,
    capital: float,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    fee_rate: float = DEFAULT_FEE,
) -> PortfolioResult:
    """Benchmark MASI buy-and-hold (frais d'entrée uniques, pas d'actions)."""
    df = masi_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    entry_date = pd.Timestamp(entry_date).normalize()
    exit_date = pd.Timestamp(exit_date).normalize()

    start_rows = df[df["date"] <= entry_date]
    end_rows = df[df["date"] <= exit_date]
    if start_rows.empty or end_rows.empty:
        return PortfolioResult(
            name="MASI",
            capital=capital,
            cash=capital,
            fee_total=0.0,
            value_final=capital,
            pnl=0.0,
            total_return=0.0,
            wealth=pd.Series(dtype=float),
            notes=["Série MASI indisponible sur la période."],
            skipped=["MASI"],
        )

    d0 = start_rows.iloc[-1]["date"]
    level0 = float(start_rows.iloc[-1]["wealth100"])
    window = df[(df["date"] >= d0) & (df["date"] <= exit_date)].copy()
    fee = capital * fee_rate
    net = capital - fee
    wealth = net * (window["wealth100"].astype(float) / level0)
    wealth.index = pd.to_datetime(window["date"])
    value_final = float(wealth.iloc[-1]) if not wealth.empty else net
    pnl = value_final - capital
    total_return = value_final / capital - 1.0 if capital > 0 else 0.0

    rets = wealth.pct_change().dropna()
    metrics = (
        compute_backtest_metrics(rets, pd.Series(0.0, index=rets.index), name="MASI")
        if len(rets) >= 2
        else {}
    )
    if metrics:
        metrics["total_return"] = round(total_return, 4)

    return PortfolioResult(
        name="MASI",
        capital=capital,
        cash=0.0,
        fee_total=fee,
        value_final=value_final,
        pnl=pnl,
        total_return=total_return,
        wealth=wealth,
        metrics=metrics,
        notes=[
            "Benchmark indiciel MASI buy-and-hold.",
            f"Frais d'entrée notionnels {100 * fee_rate:.2f} % (comparable).",
        ],
    )


def wealth_to_base100(wealth: pd.Series) -> pd.Series:
    if wealth is None or wealth.empty:
        return pd.Series(dtype=float)
    base = float(wealth.iloc[0])
    if base == 0 or not np.isfinite(base):
        return pd.Series(dtype=float)
    return 100.0 * wealth / base


def ia_buy_equal_weights(titles_asof: pd.DataFrame) -> dict[str, float]:
    buys = titles_asof[titles_asof["recommendation"].astype(str).str.upper() == "BUY"]
    return equal_weights(buys["ticker"].astype(str).tolist())


def ia_nsga_weights(holdings_asof: pd.DataFrame) -> dict[str, float]:
    if holdings_asof.empty:
        return {}
    h = holdings_asof.copy()
    h["ticker"] = h["ticker"].astype(str)
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce").fillna(0.0)
    h = h[h["weight"] > 0]
    return normalize_weights(dict(zip(h["ticker"], h["weight"])))


def equal_weight_nonsell(titles_asof: pd.DataFrame) -> dict[str, float]:
    """Univers équipondéré hors SELL (aligné backtests EW)."""
    uni = titles_asof[titles_asof["recommendation"].astype(str).str.upper() != "SELL"]
    return equal_weights(uni["ticker"].astype(str).tolist())


def run_simulation(
    *,
    capital: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    user_weights: dict[str, float],
    titles_month: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    masi_long: pd.DataFrame,
    calendar: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE,
) -> dict[str, Any]:
    """Orchestre la simulation utilisateur + 4 références."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if capital <= 0:
        raise ValueError("Le montant initial doit être > 0.")
    if start >= end:
        raise ValueError("La date de début doit être strictement antérieure à la date de fin.")
    if start < MIN_SIM_START - pd.Timedelta(days=31):
        raise ValueError("La simulation commence au plus tôt en juillet 2018.")

    month, entry_date, asof_note = resolve_as_of(start, calendar)
    if entry_date >= end:
        raise ValueError(
            f"Date d'entrée effective ({entry_date.date()}) ≥ date de fin. "
            "Élargissez la période."
        )

    titles_asof = titles_month[titles_month["month"].astype(str) == month].copy()
    if titles_asof.empty:
        raise ValueError(f"Aucune recommandation pour le mois {month}.")

    hold_asof = holdings[holdings["month"].astype(str) == month].copy()
    name_map = dict(zip(titles_asof["ticker"].astype(str), titles_asof["name"].astype(str)))
    reco_map = dict(
        zip(titles_asof["ticker"].astype(str), titles_asof["recommendation"].astype(str))
    )

    # MASI daily returns for Sharpe excess metrics
    masi = masi_long.copy()
    if not masi.empty:
        masi["date"] = pd.to_datetime(masi["date"])
        masi = masi.sort_values("date").set_index("date")
        masi_rets = masi["ret"] if "ret" in masi.columns else masi["wealth100"].pct_change()
    else:
        masi_rets = pd.Series(dtype=float)

    portfolios: dict[str, PortfolioResult] = {}
    portfolios["user"] = build_buy_hold(
        name="Mon portefeuille",
        capital=capital,
        weights=user_weights,
        prices=prices,
        entry_date=entry_date,
        exit_date=end,
        fee_rate=fee_rate,
        names=name_map,
        recommendations=reco_map,
        masi_returns=masi_rets,
    )

    w_ia_buy = ia_buy_equal_weights(titles_asof)
    portfolios["ia_buy_ew"] = build_buy_hold(
        name="IA — BUY équipondéré",
        capital=capital,
        weights=w_ia_buy,
        prices=prices,
        entry_date=entry_date,
        exit_date=end,
        fee_rate=fee_rate,
        names=name_map,
        recommendations=reco_map,
        masi_returns=masi_rets,
    )

    w_nsga = ia_nsga_weights(hold_asof)
    portfolios["ia_nsga"] = build_buy_hold(
        name="IA — NSGA-III C2",
        capital=capital,
        weights=w_nsga,
        prices=prices,
        entry_date=entry_date,
        exit_date=end,
        fee_rate=fee_rate,
        names=name_map,
        recommendations=reco_map,
        masi_returns=masi_rets,
    )

    w_ew = equal_weight_nonsell(titles_asof)
    portfolios["equal_weight"] = build_buy_hold(
        name="Equal Weight (hors SELL)",
        capital=capital,
        weights=w_ew,
        prices=prices,
        entry_date=entry_date,
        exit_date=end,
        fee_rate=fee_rate,
        names=name_map,
        recommendations=reco_map,
        masi_returns=masi_rets,
    )

    portfolios["masi"] = masi_buy_hold_wealth(
        masi_long,
        capital=capital,
        entry_date=entry_date,
        exit_date=end,
        fee_rate=fee_rate,
    )

    return {
        "month": month,
        "entry_date": entry_date,
        "exit_date": end,
        "asof_note": asof_note,
        "fee_rate": fee_rate,
        "capital": capital,
        "titles_asof": titles_asof,
        "portfolios": portfolios,
        "disclaimer": (
            "Cette simulation repose sur des données historiques et ne constitue "
            "pas une garantie de performance future."
        ),
    }
