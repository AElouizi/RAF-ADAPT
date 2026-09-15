"""Orchestrator — séquence Features → Selection → Allocation → Backtest → Evaluation."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from bvc_recommender.agents import memory
from bvc_recommender.agents.allocation_agent import AllocationAgent
from bvc_recommender.agents.backtest_agent import BacktestAgent
from bvc_recommender.agents.contracts import (
    PIPELINE_ORDER,
    AgentName,
    JobStatus,
    WorkflowRequest,
)
from bvc_recommender.agents.evaluation_agent import EvaluationAgent
from bvc_recommender.agents.feature_agent import FeatureAgent
from bvc_recommender.agents.scientific_guard import scientific_parameters
from bvc_recommender.agents.selection_agent import SelectionAgent

logger = logging.getLogger(__name__)

AGENTS = {
    AgentName.FEATURE: FeatureAgent(),
    AgentName.SELECTION: SelectionAgent(),
    AgentName.ALLOCATION: AllocationAgent(),
    AgentName.BACKTEST: BacktestAgent(),
    AgentName.EVALUATION: EvaluationAgent(),
}


def run_workflow(request: WorkflowRequest | None = None) -> dict:
    request = request or WorkflowRequest()
    memory.set_local_only(not request.write_supabase)
    workflow_id = request.workflow_id or str(uuid.uuid4())
    configuration = ",".join(f"C{c}" for c in request.cells)
    orch_job = memory.create_job(
        agent_name=AgentName.ORCHESTRATOR.value,
        task_type="workflow",
        workflow_id=workflow_id,
        period_start=request.period_start,
        period_end=request.period_end,
        configuration=configuration,
        metadata={
            "mode": request.mode.value,
            "scientific_parameters": scientific_parameters(),
        },
    )
    memory.update_job(orch_job, status=JobStatus.RUNNING.value, started_at=_iso())
    memory.add_log(
        orch_job,
        workflow_id,
        AgentName.ORCHESTRATOR.value,
        "INFO",
        f"Workflow {request.mode.value} {request.period_start}→{request.period_end} {configuration}",
    )

    start_idx = 0
    if request.resume_from:
        names = [a.value for a in PIPELINE_ORDER]
        if request.resume_from not in names:
            memory.update_job(
                orch_job,
                status=JobStatus.FAILED.value,
                completed_at=_iso(),
                error_message=f"resume_from inconnu : {request.resume_from}",
            )
            return {"workflow_id": workflow_id, "status": JobStatus.FAILED.value}
        start_idx = names.index(request.resume_from)

    results = {}
    for agent_name in PIPELINE_ORDER[start_idx:]:
        job_id = memory.create_job(
            agent_name=agent_name.value,
            task_type=agent_name.value,
            workflow_id=workflow_id,
            period_start=request.period_start,
            period_end=request.period_end,
            configuration=configuration,
            metadata={"mode": request.mode.value},
        )
        memory.add_log(
            orch_job,
            workflow_id,
            AgentName.ORCHESTRATOR.value,
            "INFO",
            f"Déclenche {agent_name.value} (job {job_id})",
        )
        result = AGENTS[agent_name].run(job_id, workflow_id, request)
        results[agent_name.value] = {
            "status": result.status.value,
            "error": result.error_message,
            "rows": result.rows_written,
            "job_id": job_id,
        }
        if result.status != JobStatus.SUCCESS:
            memory.emit_event(
                workflow_id=workflow_id,
                job_id=orch_job,
                event_type="WORKFLOW_FAILED",
                producer_agent=AgentName.ORCHESTRATOR.value,
                payload={"failed_agent": agent_name.value, "error": result.error_message},
            )
            memory.update_job(
                orch_job,
                status=JobStatus.FAILED.value,
                completed_at=_iso(),
                error_message=f"{agent_name.value}: {result.error_message}",
            )
            memory.add_log(
                orch_job,
                workflow_id,
                AgentName.ORCHESTRATOR.value,
                "ERROR",
                f"Stop : {agent_name.value} FAILED — étapes suivantes non lancées",
            )
            return {
                "workflow_id": workflow_id,
                "status": JobStatus.FAILED.value,
                "failed_agent": agent_name.value,
                "results": results,
            }

    memory.emit_event(
        workflow_id=workflow_id,
        job_id=orch_job,
        event_type="WORKFLOW_SUCCESS",
        producer_agent=AgentName.ORCHESTRATOR.value,
        payload={"results": {k: v["status"] for k, v in results.items()}},
    )
    memory.update_job(orch_job, status=JobStatus.SUCCESS.value, completed_at=_iso())
    memory.add_log(orch_job, workflow_id, AgentName.ORCHESTRATOR.value, "INFO", "Workflow SUCCESS")
    return {"workflow_id": workflow_id, "status": JobStatus.SUCCESS.value, "results": results}


def retry_failed_step(workflow_id: str, agent_name: str, request: WorkflowRequest) -> dict:
    request.workflow_id = workflow_id
    request.resume_from = agent_name
    return run_workflow(request)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()
