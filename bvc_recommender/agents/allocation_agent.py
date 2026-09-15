"""Portfolio Allocation Agent — Stage 2 validé (NSGA-III 3 obj, knee, w_i ≤ 10 %)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from bvc_recommender.agents import memory
from bvc_recommender.agents.base import BaseAgent
from bvc_recommender.agents.contracts import (
    CELL_LABELS,
    MODEL_VERSION,
    AgentName,
    AgentResult,
    JobStatus,
    WorkflowMode,
    WorkflowRequest,
)
from bvc_recommender.agents.scientific_guard import (
    AGENT_RUNS_DIR,
    FINAL_DIR,
    refuse_recompute_frozen,
    scientific_parameters,
    validated_stage2_config,
)
from bvc_recommender.config import TABLE_AGENT_PORTFOLIO_HOLDINGS
from bvc_recommender.models.portfolio_optimizer import PORTFOLIO_MAX_WEIGHT


class AllocationAgent(BaseAgent):
    name = AgentName.ALLOCATION

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
                error_message=(
                    "COMPUTE_NEW Allocation hors fenêtre figée : lancer manuellement "
                    "run_stage2_nsga.py avec --liquidity-mode pareto --w-max 0.10 "
                    "(paramètres scientifiques inchangés)."
                ),
            )
        return self._ingest_frozen(workflow_id, request)

    def _ingest_frozen(self, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        cfg = validated_stage2_config()
        frames = []
        for cell in request.cells:
            path = FINAL_DIR / "04_final_portfolios" / f"C{cell}_holdings_knee.csv"
            if not path.is_file():
                return AgentResult(
                    status=JobStatus.FAILED,
                    error_message=f"Holdings figés absents : {path}",
                )
            h = pd.read_csv(path)
            h["month"] = h["month"].astype(str)
            h = h[(h["month"] >= request.period_start) & (h["month"] <= request.period_end)]
            h["cell"] = int(cell)
            h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
            frames.append(h)
        hold = pd.concat(frames, ignore_index=True)
        wmax = float(hold["weight"].max()) if not hold.empty else 0.0
        if wmax > float(PORTFOLIO_MAX_WEIGHT) + 1e-6:
            return AgentResult(
                status=JobStatus.FAILED,
                error_message=f"Contrainte w_i≤10% violée dans le livrable (max={wmax}).",
            )
        produced = datetime.now(timezone.utc).isoformat()
        params = scientific_parameters()
        hold["run_id"] = workflow_id
        hold["agent_name"] = self.name.value
        hold["configuration"] = hold["cell"].map(lambda c: CELL_LABELS.get(int(c), f"C{c}"))
        hold["ticker"] = hold["ticker"].astype(str)
        hold["recommendation"] = hold["recommendation"].astype(str).str.upper()
        hold["predicted_return"] = pd.to_numeric(hold["predicted_return"], errors="coerce")
        hold["date_rebalance"] = pd.to_datetime(hold["date_rebalance"], errors="coerce").dt.date.astype(str)
        hold["model_version"] = MODEL_VERSION
        hold["parameters"] = json.dumps(params, ensure_ascii=False)
        hold["produced_at"] = produced
        hold["status"] = JobStatus.SUCCESS.value
        cols = [
            "run_id",
            "agent_name",
            "cell",
            "configuration",
            "month",
            "date_rebalance",
            "ticker",
            "weight",
            "recommendation",
            "predicted_return",
            "model_version",
            "parameters",
            "produced_at",
            "status",
        ]
        out = hold[cols]
        dest = AGENT_RUNS_DIR / workflow_id / "allocation"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "holdings.parquet"
        out.to_parquet(path, index=False)
        pub = memory.upsert_dataframe(TABLE_AGENT_PORTFOLIO_HOLDINGS, out, run_id=workflow_id)
        memory.register_artifact(
            run_id=workflow_id,
            agent_name=self.name.value,
            configuration=",".join(f"C{c}" for c in request.cells),
            period_start=request.period_start,
            period_end=request.period_end,
            model_version=MODEL_VERSION,
            parameters={**params, "stage2_liquidity_mode": cfg.liquidity_mode, "w_max": cfg.w_max},
            artifact_path=str(path),
            checksum_sha256=None,
            status="SUCCESS",
        )
        return AgentResult(
            status=JobStatus.SUCCESS,
            event_type="ALLOCATION_READY",
            rows_written=int(len(out)),
            artifacts={"holdings": str(path)},
            metadata={"publish": pub, "weight_max": wmax, "nsga": "NSGA-III knee (figé)"},
        )
