"""Agent de base : statut, logs, événement de sortie — pas d'appel direct inter-agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from bvc_recommender.agents import memory
from bvc_recommender.agents.contracts import (
    READY_EVENT,
    AgentName,
    AgentResult,
    JobStatus,
    WorkflowRequest,
)
from bvc_recommender.agents.scientific_guard import scientific_parameters

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: AgentName

    def run(self, job_id: str, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        memory.update_job(job_id, status=JobStatus.RUNNING.value, started_at=_iso())
        memory.add_log(
            job_id,
            workflow_id,
            self.name.value,
            "INFO",
            f"Démarrage {self.name.value} mode={request.mode.value}",
            {"cells": list(request.cells), "period": [request.period_start, request.period_end]},
        )
        try:
            result = self.execute(job_id, workflow_id, request)
        except Exception as exc:
            logger.exception("%s failed", self.name.value)
            result = AgentResult(
                status=JobStatus.FAILED,
                error_message=str(exc),
            )
        if result.status == JobStatus.SUCCESS:
            event = result.event_type or READY_EVENT.get(self.name)
            memory.update_job(
                job_id,
                status=JobStatus.SUCCESS.value,
                completed_at=_iso(),
                error_message=None,
                metadata={
                    "rows_written": result.rows_written,
                    "artifacts": result.artifacts,
                    **(result.metadata or {}),
                    "scientific_parameters": scientific_parameters(),
                },
            )
            if event:
                memory.emit_event(
                    workflow_id=workflow_id,
                    job_id=job_id,
                    event_type=event,
                    producer_agent=self.name.value,
                    payload={"rows_written": result.rows_written, "artifacts": result.artifacts},
                )
            memory.add_log(job_id, workflow_id, self.name.value, "INFO", f"SUCCESS {event}")
        else:
            memory.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=_iso(),
                error_message=result.error_message,
            )
            memory.emit_event(
                workflow_id=workflow_id,
                job_id=job_id,
                event_type="AGENT_FAILED",
                producer_agent=self.name.value,
                payload={"error": result.error_message},
            )
            memory.add_log(
                job_id,
                workflow_id,
                self.name.value,
                "ERROR",
                result.error_message or "échec",
            )
        return result

    @abstractmethod
    def execute(self, job_id: str, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        raise NotImplementedError


def _iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
