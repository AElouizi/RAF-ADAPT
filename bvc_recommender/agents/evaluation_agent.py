"""Evaluation Agent — Sharpe/Sortino/CVaR/MDD et tableaux C1–C4 vs benchmarks (figés)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from bvc_recommender.agents import memory
from bvc_recommender.agents.base import BaseAgent
from bvc_recommender.agents.contracts import (
    MODEL_VERSION,
    AgentName,
    AgentResult,
    JobStatus,
    WorkflowMode,
    WorkflowRequest,
)
from bvc_recommender.agents.scientific_guard import (
    AGENT_RUNS_DIR,
    MART_DIR,
    refuse_recompute_frozen,
    scientific_parameters,
)
from bvc_recommender.config import TABLE_AGENT_EVALUATION_KPIS


class EvaluationAgent(BaseAgent):
    name = AgentName.EVALUATION

    def execute(self, job_id: str, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        if request.mode == WorkflowMode.COMPUTE_NEW:
            refuse_recompute_frozen(
                mode=request.mode.value,
                allow_overwrite=request.allow_overwrite_historical,
                period_start=request.period_start,
                period_end=request.period_end,
            )
            return AgentResult(
                status=JobStatus.FAILED,
                error_message="COMPUTE_NEW Evaluation hors livrable figé non activé (kpi_C1C4.csv source de vérité).",
            )
        return self._ingest_frozen(workflow_id, request)

    def _ingest_frozen(self, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        kpi_path = MART_DIR / "kpi_C1C4.csv"
        if not kpi_path.is_file():
            return AgentResult(status=JobStatus.FAILED, error_message=f"KPI figés absents : {kpi_path}")
        kpis = pd.read_csv(kpi_path)
        produced = datetime.now(timezone.utc).isoformat()
        params = scientific_parameters()
        wanted = []
        labels = {1: "C1 Ridge", 2: "C2 Ridge + régime", 3: "C3 Hybrid", 4: "C4 Hybrid + régime"}
        for cell in request.cells:
            wanted.append(labels[cell])
        # garder aussi MASI et EW des cellules demandées
        mask = kpis["Strategie"].astype(str).isin(wanted + ["MASI"])
        for cell in request.cells:
            mask = mask | kpis["Strategie"].astype(str).str.contains(rf"EqualWeight C{cell}", regex=True)
        sub = kpis[mask].copy()
        out = pd.DataFrame(
            {
                "run_id": workflow_id,
                "agent_name": self.name.value,
                "configuration": sub["Strategie"].astype(str),
                "strategie": sub["Strategie"].astype(str),
                "n_months": pd.to_numeric(sub.get("n_months"), errors="coerce"),
                "rendement_ann": pd.to_numeric(sub.get("Rendement"), errors="coerce"),
                "volatilite": pd.to_numeric(sub.get("Volatilite"), errors="coerce"),
                "sharpe": pd.to_numeric(sub.get("Sharpe"), errors="coerce"),
                "sortino": pd.to_numeric(sub.get("Sortino"), errors="coerce"),
                "cvar": pd.to_numeric(sub.get("CVaR"), errors="coerce"),
                "max_dd": pd.to_numeric(sub.get("Max_DD"), errors="coerce"),
                "turnover": pd.to_numeric(sub.get("Turnover"), errors="coerce"),
                "liquidite": pd.to_numeric(sub.get("Liquidite"), errors="coerce"),
                "wealth_final_100": pd.to_numeric(sub.get("wealth_final_100"), errors="coerce"),
                "model_version": MODEL_VERSION,
                "parameters": json.dumps(params, ensure_ascii=False),
                "produced_at": produced,
                "status": JobStatus.SUCCESS.value,
            }
        )
        dest = AGENT_RUNS_DIR / workflow_id / "evaluation"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "kpis.parquet"
        out.to_parquet(path, index=False)
        extras = {}
        for name in ("tests_statistiques.csv", "subperiods_C1C4.csv", "kpi_benchmarks.csv"):
            p = MART_DIR / name
            if p.is_file():
                extras[name] = str(p)
        pub = memory.upsert_dataframe(TABLE_AGENT_EVALUATION_KPIS, out, run_id=workflow_id)
        memory.register_artifact(
            run_id=workflow_id,
            agent_name=self.name.value,
            configuration=",".join(f"C{c}" for c in request.cells),
            period_start=request.period_start,
            period_end=request.period_end,
            model_version=MODEL_VERSION,
            parameters=params,
            artifact_path=str(path),
            checksum_sha256=None,
            status="SUCCESS",
        )
        return AgentResult(
            status=JobStatus.SUCCESS,
            event_type="EVALUATION_READY",
            rows_written=int(len(out)),
            artifacts={"kpis": str(path), **extras},
            metadata={"publish": pub, "source": str(kpi_path)},
        )
