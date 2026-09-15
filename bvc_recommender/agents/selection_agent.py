"""Stock Selection Agent — C1–C4 inchangés ; publication append-only."""

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
    MART_DIR,
    refuse_recompute_frozen,
    scientific_parameters,
    stage1_reco_path,
)
from bvc_recommender.config import TABLE_AGENT_RECOMMENDATIONS


class SelectionAgent(BaseAgent):
    name = AgentName.SELECTION

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
                    "COMPUTE_NEW Selection : réentraînement C1–C4 hors fenêtre figée "
                    "non branché ici (préservation scientifique). Utiliser ingest_frozen."
                ),
            )
        return self._ingest_frozen(job_id, workflow_id, request)

    def _ingest_frozen(self, job_id: str, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        rec = _load_recommendations(request.cells)
        if rec.empty:
            return AgentResult(
                status=JobStatus.FAILED,
                error_message="Aucune recommandation figée trouvée (parquet Stage 1 ou mart).",
            )
        rec = rec[
            (rec["month"] >= request.period_start) & (rec["month"] <= request.period_end)
        ].copy()
        rec = rec[rec["cell"].isin(request.cells)].copy()
        if rec.empty:
            return AgentResult(
                status=JobStatus.FAILED,
                error_message="Filtre période/cellules vide.",
            )

        produced = datetime.now(timezone.utc).isoformat()
        params = scientific_parameters()
        rec["run_id"] = workflow_id
        rec["agent_name"] = self.name.value
        rec["model_version"] = MODEL_VERSION
        rec["produced_at"] = produced
        rec["status"] = JobStatus.SUCCESS.value
        rec["parameters"] = json.dumps(params, ensure_ascii=False)
        rec["configuration"] = rec["cell"].map(lambda c: CELL_LABELS.get(int(c), f"C{c}"))

        keep = [
            "run_id",
            "agent_name",
            "cell",
            "configuration",
            "month",
            "date_decision",
            "ticker",
            "recommendation",
            "score",
            "conviction_score",
            "tau",
            "model_version",
            "parameters",
            "produced_at",
            "status",
        ]
        out = rec[keep].copy()
        dest = AGENT_RUNS_DIR / workflow_id / "selection"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "recommendations.parquet"
        out.to_parquet(path, index=False)
        pub = memory.upsert_dataframe(TABLE_AGENT_RECOMMENDATIONS, out, run_id=workflow_id)
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
            event_type="SELECTION_READY",
            rows_written=int(len(out)),
            artifacts={"recommendations": str(path)},
            metadata={"publish": pub, "cells": sorted(out["cell"].unique().tolist())},
        )


def _load_recommendations(cells: tuple[int, ...]) -> pd.DataFrame:
    path = stage1_reco_path()
    if path is not None:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df.get("date", df.get("date_cours")), errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["cell"] = pd.to_numeric(df["cell"], errors="coerce").astype(int)
        df["ticker"] = df["ticker"].astype(str)
        df["recommendation"] = df["recommendation"].astype(str).str.upper()
        score_col = "predicted_return" if "predicted_return" in df.columns else "prediction"
        df["score"] = pd.to_numeric(df[score_col], errors="coerce")
        df["conviction_score"] = pd.to_numeric(
            df.get("conviction_score", df["score"]), errors="coerce"
        )
        df["tau"] = pd.to_numeric(df["tau"], errors="coerce") if "tau" in df.columns else None
        df["date_decision"] = df["date"].dt.date.astype(str)
        return df

    frames = []
    mart = MART_DIR / "fact_recommendations_C4.csv"
    if mart.is_file():
        m = pd.read_csv(mart)
        m["month"] = m["month"].astype(str)
        m["cell"] = pd.to_numeric(m["cell"], errors="coerce").astype(int)
        m["ticker"] = m["ticker"].astype(str)
        m["recommendation"] = m["recommendation"].astype(str).str.upper()
        m["score"] = pd.to_numeric(m["alpha"], errors="coerce")
        m["conviction_score"] = pd.to_numeric(m["conviction_score"], errors="coerce")
        m["tau"] = pd.to_numeric(m["tau"], errors="coerce") if "tau" in m.columns else None
        m["date_decision"] = pd.to_datetime(m["date"], errors="coerce").dt.date.astype(str)
        frames.append(m)

    for cell in cells:
        hp = FINAL_DIR / "04_final_portfolios" / f"C{cell}_holdings_knee.csv"
        if not hp.is_file():
            continue
        h = pd.read_csv(hp)
        h["month"] = h["month"].astype(str)
        h["cell"] = int(cell)
        h["ticker"] = h["ticker"].astype(str)
        h["recommendation"] = h["recommendation"].astype(str).str.upper()
        h["score"] = pd.to_numeric(h["predicted_return"], errors="coerce")
        h["conviction_score"] = pd.to_numeric(h["conviction_score"], errors="coerce")
        h["tau"] = None
        h["date_decision"] = pd.to_datetime(h["date_rebalance"], errors="coerce").dt.date.astype(str)
        frames.append(h)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["cell", "month", "ticker"], keep="first")
    return out
