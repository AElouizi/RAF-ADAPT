"""Onglet Administration / Agents — non visible pour l'investisseur."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from bvc_recommender.agents import memory
from bvc_recommender.agents.contracts import WorkflowMode, WorkflowRequest
from bvc_recommender.agents.orchestrator import retry_failed_step, run_workflow
from bvc_recommender.config import load_env_file
from bvc_recommender.config import PROJECT_ROOT


def _expected_token() -> str:
    load_env_file(PROJECT_ROOT / ".env")
    return (os.getenv("BVC_ADMIN_TOKEN") or "").strip()


def render_login() -> bool:
    expected = _expected_token()
    with st.sidebar.expander("Administration", expanded=False):
        st.caption("Réservé à l'opérateur de la thèse. L'investisseur n'utilise pas cet espace.")
        token = st.text_input("Jeton", type="password", key="bvc_admin_token")
        if st.button("Ouvrir l'administration", key="bvc_admin_login"):
            ok = bool(expected) and token == expected
            if not expected and token == "these-admin":
                ok = True
                st.warning("BVC_ADMIN_TOKEN absent du .env — jeton local de secours accepté.")
            st.session_state["is_admin"] = ok
            if not ok:
                st.error("Accès refusé.")
        if st.session_state.get("is_admin"):
            st.success("Session administrateur active")
            if st.button("Fermer la session", key="bvc_admin_logout"):
                st.session_state["is_admin"] = False
                st.rerun()
    return bool(st.session_state.get("is_admin"))


def render_admin() -> None:
    st.markdown(
        '<div class="section-title">Administration — orchestration multi-agents</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Cet onglet pilote les agents (Feature, Sélection C1–C4, Allocation NSGA-III, "
        "Backtest, Évaluation). Les onglets investisseur restent en lecture seule sur le livrable figé."
    )
    jobs = memory.list_jobs(limit=40)
    last = "—"
    if not jobs.empty and "agent_name" in jobs.columns:
        orch = jobs[jobs["agent_name"] == "orchestrator"]
        if not orch.empty:
            row = orch.iloc[0]
            last = f"{row.get('created_at', '—')} · {row.get('status', '—')}"
    st.metric("Dernière exécution orchestrateur", last)

    c1, c2, c3 = st.columns(3)
    with c1:
        period_start = st.text_input("Période début (YYYY-MM)", value="2018-07")
        period_end = st.text_input("Période fin (YYYY-MM)", value="2025-06")
    with c2:
        cell_choice = st.multiselect(
            "Cellules",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: {1: "C1", 2: "C2", 3: "C3", 4: "C4"}[x],
        )
        mode = st.selectbox(
            "Mode",
            options=["ingest_frozen", "compute_new"],
            index=0,
            help="ingest_frozen : pas de recalcul scientifique. compute_new est bloqué sur 2018-07→2025-06.",
        )
    with c3:
        resume = st.selectbox(
            "Relancer depuis (optionnel)",
            options=["", "feature", "selection", "allocation", "backtest", "evaluation"],
        )
        wf_id = st.text_input("workflow_id (reprise)", value="")

    if st.button("Lancer le workflow", type="primary"):
        if not cell_choice:
            st.error("Choisir au moins une cellule.")
        else:
            req = WorkflowRequest(
                mode=WorkflowMode(mode),
                period_start=period_start.strip(),
                period_end=period_end.strip(),
                cells=tuple(sorted(cell_choice)),
                resume_from=resume or None,
                workflow_id=wf_id.strip() or None,
            )
            with st.spinner("Orchestrateur en cours…"):
                try:
                    if resume and wf_id.strip():
                        out = retry_failed_step(wf_id.strip(), resume, req)
                    else:
                        out = run_workflow(req)
                    st.session_state["last_workflow"] = out
                    if out.get("status") == "SUCCESS":
                        st.success(f"SUCCESS · workflow {out.get('workflow_id')}")
                    else:
                        st.error(
                            f"FAILED · {out.get('failed_agent')} · {out.get('workflow_id')}"
                        )
                    st.json(out)
                except Exception as exc:
                    st.exception(exc)

    st.markdown("**État des jobs**")
    jobs = memory.list_jobs(limit=80)
    if jobs.empty:
        st.info("Aucun job. Lancez un workflow ou exécutez `py -m bvc_recommender.agents`.")
    else:
        show = jobs.copy()
        keep = [
            c
            for c in (
                "created_at",
                "agent_name",
                "task_type",
                "configuration",
                "period_start",
                "period_end",
                "status",
                "error_message",
                "workflow_id",
                "id",
            )
            if c in show.columns
        ]
        st.dataframe(show[keep], use_container_width=True, hide_index=True, height=320)

        failed = show[show["status"] == "FAILED"] if "status" in show.columns else pd.DataFrame()
        if not failed.empty:
            st.markdown("**Erreurs**")
            st.dataframe(
                failed[keep],
                use_container_width=True,
                hide_index=True,
            )

    last_wf = st.session_state.get("last_workflow") or {}
    wf_filter = st.text_input(
        "Logs — workflow_id",
        value=str(last_wf.get("workflow_id") or ""),
    )
    logs = memory.list_logs(workflow_id=wf_filter or None, limit=200)
    st.markdown("**Logs**")
    if logs.empty:
        st.caption("Pas de logs.")
    else:
        lkeep = [c for c in ("created_at", "agent_name", "level", "message", "workflow_id") if c in logs.columns]
        st.dataframe(logs[lkeep], use_container_width=True, hide_index=True, height=280)

    events = memory.list_events(workflow_id=wf_filter or None)
    st.markdown("**Événements**")
    if events.empty:
        st.caption("Pas d'événements locaux.")
    else:
        ekeep = [c for c in ("created_at", "event_type", "producer_agent", "workflow_id") if c in events.columns]
        st.dataframe(events[ekeep], use_container_width=True, hide_index=True, height=200)

    st.caption(
        "SQL tables : `bvc_recommender/scripts/sql/create_agent_tables.sql`. "
        "Alimentation marché / fondamentaux = responsabilité administrateur (tables existantes)."
    )
