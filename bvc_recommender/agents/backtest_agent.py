"""Backtesting Agent — protocole existant (fichiers 05_backtest), append-only."""

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
)
from bvc_recommender.config import TABLE_AGENT_BACKTEST_MONTHLY


class BacktestAgent(BaseAgent):
    name = AgentName.BACKTEST

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
                    "COMPUTE_NEW Backtest hors fenêtre figée : réutiliser "
                    "assemble_final_experiment.backtest_holdings (protocole inchangé)."
                ),
            )
        return self._ingest_frozen(workflow_id, request)

    def _ingest_frozen(self, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        frames = []
        daily_paths = {}
        for cell in request.cells:
            monthly_p = FINAL_DIR / "05_backtest" / f"C{cell}_monthly.csv"
            daily_p = FINAL_DIR / "05_backtest" / f"C{cell}_daily.csv"
            if not monthly_p.is_file():
                return AgentResult(
                    status=JobStatus.FAILED,
                    error_message=f"Backtest mensuel figé absent : {monthly_p}",
                )
            m = pd.read_csv(monthly_p)
            m["month"] = m["month"].astype(str)
            m = m[(m["month"] >= request.period_start) & (m["month"] <= request.period_end)]
            m["cell"] = int(cell)
            frames.append(m)
            if daily_p.is_file():
                daily_paths[f"C{cell}"] = str(daily_p)
        bt = pd.concat(frames, ignore_index=True)
        produced = datetime.now(timezone.utc).isoformat()
        params = scientific_parameters()
        parts = []
        for _cell, g in bt.groupby("cell"):
            g = g.sort_values("month").copy()
            w = 100.0
            ws = []
            for r in g.itertuples(index=False):
                w *= 1.0 + float(getattr(r, "net_return") or 0.0)
                ws.append(w)
            g["wealth100"] = ws
            parts.append(g)
        bt = pd.concat(parts, ignore_index=True)
        bt["run_id"] = workflow_id
        bt["agent_name"] = self.name.value
        bt["configuration"] = bt["cell"].map(lambda c: CELL_LABELS.get(int(c), f"C{c}"))
        bt["model_version"] = MODEL_VERSION
        bt["parameters"] = json.dumps(params, ensure_ascii=False)
        bt["produced_at"] = produced
        bt["status"] = JobStatus.SUCCESS.value
        cols = [
            "run_id",
            "agent_name",
            "cell",
            "configuration",
            "month",
            "net_return",
            "gross_return",
            "turnover",
            "transaction_cost",
            "n_positions",
            "wealth100",
            "model_version",
            "parameters",
            "produced_at",
            "status",
        ]
        out = bt[cols]
        dest = AGENT_RUNS_DIR / workflow_id / "backtest"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "monthly.parquet"
        out.to_parquet(path, index=False)
        pub = memory.upsert_dataframe(TABLE_AGENT_BACKTEST_MONTHLY, out, run_id=workflow_id)
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
            event_type="BACKTEST_READY",
            rows_written=int(len(out)),
            artifacts={"monthly": str(path), **daily_paths},
            metadata={"publish": pub, "daily_frozen_pointers": daily_paths},
        )
