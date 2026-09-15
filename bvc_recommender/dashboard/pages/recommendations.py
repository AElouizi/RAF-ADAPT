"""Onglet 1 — Recommandations (top 20 + badges NOUVEAU/SORTI vs mois précédent)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bvc_recommender.dashboard.components import html_label_badge, html_move_badge
from bvc_recommender.dashboard.data import (
    get_ranked_universe,
    get_rebalance_date,
    list_periods,
    load_fundamental_features,
    load_technical_features,
)
from bvc_recommender.rebalance import period_display_label


def _fmt_num(x, digits: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if pct:
        return f"{float(x):.1%}"
    return f"{float(x):.{digits}f}"


def _top_set(period: str) -> set[str]:
    ranked = get_ranked_universe(period)
    if ranked.empty:
        return set()
    return set(ranked.loc[ranked["label"] == "TOP", "ticker"])


def render(period: str) -> None:
    ranked = get_ranked_universe(period)
    if ranked.empty:
        st.warning(f"Données de ranking indisponibles pour {period_display_label(period)}.")
        return

    periods = list_periods()
    p_idx = periods.index(period) if period in periods else -1
    prev_period = periods[p_idx - 1] if p_idx > 0 else None
    prev_top = _top_set(prev_period) if prev_period else set()
    curr_top = set(ranked.loc[ranked["label"] == "TOP", "ticker"])

    top20 = ranked.head(20).copy()
    as_of = get_rebalance_date(period)

    tech = load_technical_features()
    fund = load_fundamental_features()

    if not tech.empty and as_of is not None:
        tech = tech.copy()
        tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
        tech = tech[tech["date_cours"] <= as_of]
        tech = tech.sort_values("date_cours").groupby("ticker", as_index=False).tail(1)
        tech_cols = [c for c in ("rsi_14", "beta_masi", "vmq_20j", "ret_3m") if c in tech.columns]
        top20 = top20.merge(tech[["ticker", *tech_cols]], on="ticker", how="left", suffixes=("", "_tech"))

    if not fund.empty and as_of is not None:
        fund = fund.copy()
        date_col = "date_fin" if "date_fin" in fund.columns else "date_cours"
        if date_col in fund.columns:
            fund[date_col] = pd.to_datetime(fund[date_col], errors="coerce")
            fund = fund[fund[date_col] <= as_of]
            fund = fund.sort_values(date_col).groupby("ticker", as_index=False).tail(1)
            fund_cols = [c for c in ("pe", "pb", "roe", "score_sante") if c in fund.columns]
            if fund_cols:
                top20 = top20.merge(
                    fund[["ticker", *fund_cols]], on="ticker", how="left", suffixes=("", "_fund")
                )

    def _badge(ticker: str) -> str:
        if ticker in curr_top and ticker not in prev_top:
            return "NOUVEAU"
        if ticker not in curr_top and ticker in prev_top:
            return "SORTI"
        return ""

    top20["badge"] = top20["ticker"].map(_badge)
    sortis_in_univ = ranked[ranked["ticker"].isin(prev_top - curr_top)].head(10)

    prev_label = period_display_label(prev_period) if prev_period else "—"
    st.markdown(
        f'<div class="section-title">Top 20 — {period_display_label(period)}'
        f' <span style="color:#64748b;font-weight:500;font-size:0.9rem">'
        f'(vs {prev_label})</span></div>',
        unsafe_allow_html=True,
    )

    rows_html = []
    for _, row in top20.iterrows():
        label_html = html_label_badge(str(row.get("label", "")))
        move_html = html_move_badge(str(row.get("badge", "")))
        vmq = row.get("vmq_20j")
        vmq_txt = f"{vmq:,.0f}" if pd.notna(vmq) else "—"
        rows_html.append(
            "<tr>"
            f"<td><b>{row['ticker']}</b></td>"
            f"<td>{label_html}</td>"
            f"<td>{_fmt_num(row.get('score_final'), 4)}</td>"
            f"<td>{_fmt_num(row.get('prediction') or row.get('score_tft'), 4)}</td>"
            f"<td>{vmq_txt}</td>"
            f"<td>{_fmt_num(row.get('pe'), 1)}</td>"
            f"<td>{_fmt_num(row.get('pb'), 2)}</td>"
            f"<td>{_fmt_num(row.get('roe'), 1)}</td>"
            f"<td>{_fmt_num(row.get('rsi_14'), 1)}</td>"
            f"<td>{_fmt_num(row.get('ret_3m'), pct=True)}</td>"
            f"<td>{move_html}</td>"
            "</tr>"
        )

    table = f"""
    <div style="overflow-x:auto;background:#fff;border:1px solid #d8e0ea;border-radius:12px;padding:0.4rem">
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
      <thead>
        <tr style="text-align:left;color:#64748b;border-bottom:1px solid #e2e8f0">
          <th style="padding:8px">Ticker</th>
          <th>Label</th>
          <th>Score final</th>
          <th>Score TFT</th>
          <th>VMQ 20j</th>
          <th>P/E</th>
          <th>P/B</th>
          <th>ROE</th>
          <th>RSI</th>
          <th>Mom 3M</th>
          <th>Mouvement</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)

    n_new = int((top20["badge"] == "NOUVEAU").sum())
    n_top = int((ranked["label"] == "TOP").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Univers", len(ranked))
    c2.metric("TOP", n_top)
    c3.metric("Entrées TOP (vs M-1)", n_new)

    if prev_period and not sortis_in_univ.empty:
        with st.expander(f"SORTI du TOP vs {prev_label}"):
            show = sortis_in_univ[["ticker", "label", "score_final"]].copy()
            show["badge"] = "SORTI"
            st.dataframe(show, hide_index=True, use_container_width=True)
