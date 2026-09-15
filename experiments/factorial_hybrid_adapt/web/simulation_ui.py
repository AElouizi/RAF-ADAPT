"""
UI Streamlit — onglet Simulation (buy-and-hold utilisateur).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys

from experiments.factorial_hybrid_adapt.web.simulation_engine import (
    DEFAULT_FEE,
    MIN_SIM_START,
    equal_weights,
    label_reco_fr,
    run_simulation,
    validate_weights_sum,
    wealth_to_base100,
)

PORTFOLIO_KEYS = [
    ("user", "Mon portefeuille", "#0c2340"),
    ("ia_buy_ew", "IA — BUY équipondéré", "#15803d"),
    ("ia_nsga", "IA — NSGA-III (champion)", "#0e7490"),
    ("equal_weight", "Equal Weight", "#b45309"),
    ("masi", "MASI", "#1d4ed8"),
]


def _data():
    """Module data_c4 courant (après reload dans app_c4)."""
    return sys.modules["experiments.factorial_hybrid_adapt.web.data_c4"]


def _fmt_mad(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):,.{digits}f} MAD".replace(",", " ")


def _fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * float(x):.{digits}f} %"


def _default_dates() -> tuple[date, date]:
    months = _data().list_months()
    start = date(2018, 7, 31)
    end = date(2025, 6, 30)
    if months:
        try:
            start = datetime.strptime(months[0] + "-01", "%Y-%m-%d").date().replace(day=31)
        except Exception:
            pass
        try:
            y, m = months[-1].split("-")
            end = (pd.Timestamp(f"{y}-{m}-01") + pd.offsets.MonthEnd(0)).date()
        except Exception:
            pass
    return start, end


def render_simulation() -> None:
    st.markdown(
        '<div class="section-title">Simulation — performance historique d’un portefeuille</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Choisissez un montant, une période (≥ juil. 2018), consultez les recommandations "
        "disponibles à la date d’entrée, sélectionnez vos titres et allouez le capital. "
        "Les recommandations IA sont indicatives : la décision finale vous appartient."
    )

    d0, d1 = _default_dates()
    with st.form("sim_params"):
        st.markdown("##### 1 — Paramètres de simulation")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 0.9])
        capital = c1.number_input(
            "Montant initial (MAD)",
            min_value=1_000.0,
            max_value=100_000_000.0,
            value=100_000.0,
            step=1_000.0,
            format="%.0f",
        )
        start = c2.date_input(
            "Date de début",
            value=d0,
            min_value=date(2018, 7, 1),
            max_value=date(2025, 12, 31),
        )
        end = c3.date_input(
            "Date de fin",
            value=d1,
            min_value=date(2018, 7, 1),
            max_value=date(2026, 12, 31),
        )
        fee_pct = c4.number_input(
            "Frais de transaction (%)",
            min_value=0.0,
            max_value=5.0,
            value=round(100 * DEFAULT_FEE, 2),
            step=0.05,
            help="Défaut 0,3 % (aligné backtests RAF-ADAPT). Appliqués à l’entrée.",
        )
        btn_load_reco = st.form_submit_button("Charger les recommandations", type="primary")

    if btn_load_reco or st.session_state.get("sim_params_ready"):
        if btn_load_reco:
            st.session_state["sim_params_ready"] = True
            st.session_state["sim_capital"] = float(capital)
            st.session_state["sim_start"] = pd.Timestamp(start)
            st.session_state["sim_end"] = pd.Timestamp(end)
            st.session_state["sim_fee"] = float(fee_pct) / 100.0
            st.session_state.pop("sim_result", None)
            st.session_state.pop("sim_selected", None)
            st.session_state.pop("sim_custom_w", None)

    if not st.session_state.get("sim_params_ready"):
        st.info("Renseignez les paramètres puis cliquez sur **Charger les recommandations**.")
        return

    capital = float(st.session_state["sim_capital"])
    start_ts = pd.Timestamp(st.session_state["sim_start"])
    end_ts = pd.Timestamp(st.session_state["sim_end"])
    fee_rate = float(st.session_state["sim_fee"])

    if capital <= 0:
        st.error("Le montant initial doit être > 0.")
        return
    if start_ts >= end_ts:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    data = _data()
    titles = data.load_titles()
    calendar = data.load_decision_calendar()
    from experiments.factorial_hybrid_adapt.web.simulation_engine import resolve_as_of

    try:
        month, entry_date, asof_note = resolve_as_of(start_ts, calendar)
    except ValueError as exc:
        st.error(str(exc))
        return

    titles_asof = titles[titles["month"].astype(str) == month].copy()
    if titles_asof.empty:
        st.error(f"Aucune recommandation disponible pour {month}.")
        return

    st.markdown("##### 2 — Recommandations des titres")
    st.caption(asof_note)
    if entry_date.normalize() != start_ts.normalize():
        st.warning(
            f"Entrée effective le **{entry_date.date()}** "
            f"(première date de décision ≥ votre date de début)."
        )

    show = titles_asof[
        ["ticker", "name", "recommendation", "alpha", "conviction_score", "in_portfolio"]
    ].copy()
    show["Reco"] = show["recommendation"].map(label_reco_fr)
    show["Alpha"] = pd.to_numeric(show["alpha"], errors="coerce").round(4)
    show["Conviction"] = pd.to_numeric(show["conviction_score"], errors="coerce").round(3)
    show["Dans NSGA"] = show["in_portfolio"].map({1: "Oui", 0: "Non"})
    show = show.sort_values(["recommendation", "alpha"], ascending=[True, False])

    # Pré-sélection : titres BUY par défaut
    default_selected = set(
        show.loc[show["recommendation"].astype(str).str.upper() == "BUY", "ticker"].astype(str)
    )
    if "sim_selected" not in st.session_state:
        st.session_state["sim_selected"] = sorted(default_selected)

    c_sel1, c_sel2, c_sel3 = st.columns(3)
    if c_sel1.button("Sélectionner tous les Achat (BUY)"):
        st.session_state["sim_selected"] = sorted(default_selected)
        st.rerun()
    if c_sel2.button("Sélectionner Achat + Neutre"):
        mask = show["recommendation"].astype(str).str.upper().isin(["BUY", "NEUTRAL"])
        st.session_state["sim_selected"] = sorted(show.loc[mask, "ticker"].astype(str))
        st.rerun()
    if c_sel3.button("Tout désélectionner"):
        st.session_state["sim_selected"] = []
        st.rerun()

    options = show["ticker"].astype(str).tolist()
    labels = {
        str(r.ticker): f"{r.ticker} — {r.name} · {label_reco_fr(r.recommendation)}"
        for r in show.itertuples()
    }
    selected = st.multiselect(
        "Titres à inclure dans mon portefeuille",
        options=options,
        default=[t for t in st.session_state.get("sim_selected", []) if t in options],
        format_func=lambda t: labels.get(t, t),
        help="Vous pouvez inclure des titres Neutre ou Vente pour tester votre propre scénario.",
    )
    st.session_state["sim_selected"] = selected

    st.dataframe(
        show[["ticker", "name", "Reco", "Alpha", "Conviction", "Dans NSGA"]].rename(
            columns={"ticker": "Ticker", "name": "Titre"}
        ),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    st.markdown("##### 3 — Construction du portefeuille (allocations)")
    if not selected:
        st.warning("Sélectionnez au moins un titre.")
        return

    mode = st.radio(
        "Mode d’allocation",
        ["Répartition équipondérée", "Allocation personnalisée"],
        horizontal=True,
    )

    if mode == "Répartition équipondérée":
        wmap = equal_weights(selected)
        alloc_df = pd.DataFrame(
            {
                "Ticker": list(wmap.keys()),
                "Poids (%)": [round(100 * wmap[t], 4) for t in wmap],
                "Montant cible (MAD)": [round(capital * wmap[t], 2) for t in wmap],
            }
        )
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        user_weights = wmap
        ok_sum = True
    else:
        n = len(selected)
        default_pct = round(100.0 / n, 4)
        edit_df = pd.DataFrame(
            {
                "Ticker": selected,
                "Poids (%)": [
                    float(
                        st.session_state.get("sim_custom_w", {}).get(t, default_pct)
                    )
                    for t in selected
                ],
            }
        )
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            disabled=["Ticker"],
            key="sim_weight_editor",
            column_config={
                "Poids (%)": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format="%.2f",
                )
            },
        )
        if st.button("Répartition équipondérée"):
            eq = equal_weights(selected)
            st.session_state["sim_custom_w"] = {t: 100.0 * eq[t] for t in eq}
            st.rerun()

        user_weights = {
            str(r["Ticker"]): float(r["Poids (%)"]) / 100.0 for _, r in edited.iterrows()
        }
        st.session_state["sim_custom_w"] = {
            k: 100.0 * v for k, v in user_weights.items()
        }
        ok_sum, raw_sum = validate_weights_sum(user_weights, tol=1e-3)
        st.caption(f"Somme des poids : **{100 * raw_sum:.2f} %** (requis : 100 %).")
        if not ok_sum:
            st.error("La somme des poids doit être égale à 100 %.")

    run = st.button("Lancer la simulation", type="primary", disabled=not ok_sum)
    if run:
        with st.spinner("Calcul de la simulation buy-and-hold…"):
            try:
                prices = data.load_price_panel()
                holdings = data.load_holdings()
                masi = data.load_masi_long()
                result = run_simulation(
                    capital=capital,
                    start=start_ts,
                    end=end_ts,
                    user_weights=user_weights,
                    titles_month=titles,
                    holdings=holdings,
                    prices=prices,
                    masi_long=masi,
                    calendar=calendar,
                    fee_rate=fee_rate,
                )
                st.session_state["sim_result"] = result
            except Exception as exc:
                st.error(f"Échec de la simulation : {exc}")
                return

    if "sim_result" not in st.session_state:
        return

    _render_results(st.session_state["sim_result"])


def _render_results(result: dict) -> None:
    portfolios = result["portfolios"]
    user = portfolios["user"]

    st.markdown("##### 4 — Performance du portefeuille")
    st.info(result["disclaimer"])
    st.caption(
        f"Mois de recommandations : **{result['month']}** · "
        f"Entrée : **{pd.Timestamp(result['entry_date']).date()}** · "
        f"Sortie : **{pd.Timestamp(result['exit_date']).date()}** · "
        f"Frais : **{100 * result['fee_rate']:.2f} %** · "
        f"{result['asof_note']}"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Capital initial", _fmt_mad(user.capital, 0))
    k2.metric("Valeur finale", _fmt_mad(user.value_final, 0))
    k3.metric("Gain / Perte", _fmt_mad(user.pnl, 0))
    k4.metric("Performance", _fmt_pct(user.total_return))
    k5.metric("Cash résiduel", _fmt_mad(user.cash, 0))

    m = user.metrics or {}
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rendement annualisé", _fmt_pct(m.get("annualized_return")))
    r2.metric("Volatilité", _fmt_pct(m.get("volatility_ann")))
    r3.metric("Sharpe", f"{m['sharpe']:.2f}" if m.get("sharpe") is not None else "—")
    r4.metric("Max Drawdown", _fmt_pct(m.get("max_drawdown")))

    st.markdown("##### Évolution (base 100)")
    fig = go.Figure()
    for key, label, color in PORTFOLIO_KEYS:
        pf = portfolios.get(key)
        if pf is None or pf.wealth is None or pf.wealth.empty:
            continue
        base = wealth_to_base100(pf.wealth)
        if base.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=base.index,
                y=base.values,
                name=label,
                mode="lines",
                line=dict(color=color, width=2.6 if key == "user" else 1.7),
            )
        )
    fig.update_layout(
        height=400,
        yaxis_title="Base 100",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dernière valeur base 100
    last_vals = []
    for key, label, _ in PORTFOLIO_KEYS:
        pf = portfolios.get(key)
        if pf is None or pf.wealth is None or pf.wealth.empty:
            continue
        b = wealth_to_base100(pf.wealth)
        if not b.empty:
            last_vals.append(f"**{label}** : {b.iloc[-1]:.1f}")
    if last_vals:
        st.caption(" · ".join(last_vals))

    st.markdown("##### Votre portefeuille vs recommandations IA")
    rows = []
    for key, label, _ in PORTFOLIO_KEYS:
        pf = portfolios.get(key)
        if pf is None:
            continue
        mm = pf.metrics or {}
        rows.append(
            {
                "Portefeuille": label,
                "Performance": _fmt_pct(pf.total_return),
                "Annualisé": _fmt_pct(mm.get("annualized_return")),
                "Volatilité": _fmt_pct(mm.get("volatility_ann")),
                "Sharpe": f"{mm['sharpe']:.2f}" if mm.get("sharpe") is not None else "—",
                "Max Drawdown": _fmt_pct(mm.get("max_drawdown")),
                "Valeur finale": _fmt_mad(pf.value_final, 0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("##### Détails de la simulation — mon portefeuille")
    if user.positions:
        det = pd.DataFrame(
            [
                {
                    "Ticker": p.ticker,
                    "Titre": p.name,
                    "Reco": label_reco_fr(p.recommendation),
                    "Poids cible (%)": round(100 * p.weight_target, 2),
                    "Prix initial": round(p.price_entry, 2),
                    "Prix final": round(p.price_exit, 2),
                    "Quantité": p.quantity,
                    "Montant investi": round(p.amount_invested, 2),
                    "Frais entrée": round(p.fee_entry, 2),
                    "Valeur finale": round(p.value_exit, 2),
                    "Gain/Perte": round(p.pnl, 2),
                    "Contribution": _fmt_pct(p.contribution),
                }
                for p in user.positions
            ]
        )
        st.dataframe(det, use_container_width=True, hide_index=True)
    else:
        st.warning("Aucune position ouverte (prix ou budget insuffisant).")

    with st.expander("Hypothèses et titres exclus"):
        for note in user.notes:
            st.write(f"• {note}")
        st.write(f"• Cash résiduel : {_fmt_mad(user.cash, 2)}")
        st.write(f"• Frais totaux : {_fmt_mad(user.fee_total, 2)}")
        if user.skipped:
            st.write("Titres / alertes :")
            for s in user.skipped:
                st.write(f"• {s}")
        st.warning(result["disclaimer"])
