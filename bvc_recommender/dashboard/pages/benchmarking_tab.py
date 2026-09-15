"""Onglet Benchmarking — tableau comparatif des 10 modèles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bvc_recommender.config import DATASET_DIR, REPORTS_DIR


def render() -> None:
    st.markdown(
        '<div class="section-title">Benchmarking & ablation — RAF-ADAPT vs alternatives</div>',
        unsafe_allow_html=True,
    )

    metrics_path = DATASET_DIR / "benchmarking" / "benchmark_metrics.csv"
    fig_path = REPORTS_DIR / "benchmarking" / "comparison_alpha_sharpe.png"

    if not metrics_path.is_file():
        st.warning(
            "Résultats absents — lancer "
            "`py -m bvc_recommender.benchmarking.run_all_benchmarks` puis "
            "`generate_comparison_table`."
        )
        return

    df = pd.read_csv(metrics_path)
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df

    show = df.copy()
    for col in ("alpha_annualized", "cvar_95", "max_drawdown", "hit_ratio"):
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ("sharpe", "sortino", "calmar", "omega"):
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    rename = {
        "model_name": "Modèle",
        "level": "Niveau",
        "alpha_annualized": "Alpha",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "cvar_95": "CVaR 95%",
        "max_drawdown": "Max DD",
        "hit_ratio": "Hit",
        "calmar": "Calmar",
        "omega": "Omega",
        "replaces": "Composante testée",
    }
    cols = [c for c in rename if c in show.columns]
    st.dataframe(
        show[cols].rename(columns=rename),
        use_container_width=True,
        hide_index=True,
    )

    plot_df = df.dropna(subset=["alpha_annualized", "sharpe"], how="any").copy()
    if not plot_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig_a = px.bar(
                plot_df,
                x="model_name",
                y="alpha_annualized",
                color="level",
                title="Alpha annualisé vs MASI",
                labels={"alpha_annualized": "Alpha", "model_name": ""},
            )
            fig_a.update_layout(xaxis_tickangle=-35, height=400)
            st.plotly_chart(fig_a, use_container_width=True)
        with c2:
            fig_s = px.bar(
                plot_df,
                x="model_name",
                y="sharpe",
                color="level",
                title="Ratio de Sharpe",
                labels={"sharpe": "Sharpe", "model_name": ""},
            )
            fig_s.update_layout(xaxis_tickangle=-35, height=400)
            st.plotly_chart(fig_s, use_container_width=True)

    if fig_path.is_file():
        st.image(str(fig_path), caption="Export article (matplotlib)")

    st.caption(f"Source : `{metrics_path}` · coûts 0,3 % · test 2023–2025 · random_state=42")
