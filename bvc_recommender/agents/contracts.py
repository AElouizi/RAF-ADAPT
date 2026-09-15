"""Contrats d'entrée / sortie des agents (couplage faible via événements)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class WorkflowMode(str, Enum):
    """ingest_frozen : publie le livrable validé, aucun recalcul scientifique."""

    INGEST_FROZEN = "ingest_frozen"
    COMPUTE_NEW = "compute_new"


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    FEATURE = "feature"
    SELECTION = "selection"
    ALLOCATION = "allocation"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"


PIPELINE_ORDER: tuple[AgentName, ...] = (
    AgentName.FEATURE,
    AgentName.SELECTION,
    AgentName.ALLOCATION,
    AgentName.BACKTEST,
    AgentName.EVALUATION,
)

READY_EVENT: dict[AgentName, str] = {
    AgentName.FEATURE: "FEATURES_READY",
    AgentName.SELECTION: "SELECTION_READY",
    AgentName.ALLOCATION: "ALLOCATION_READY",
    AgentName.BACKTEST: "BACKTEST_READY",
    AgentName.EVALUATION: "EVALUATION_READY",
}

CELL_LABELS = {
    1: "C1 Ridge",
    2: "C2 Ridge + régime",
    3: "C3 Hybrid",
    4: "C4 Hybrid + régime",
}

MODEL_VERSION = "frozen_final_201807_202506_wmax10_pareto_knee"


@dataclass
class AgentResult:
    status: JobStatus
    event_type: str | None = None
    rows_written: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class WorkflowRequest:
    mode: WorkflowMode = WorkflowMode.INGEST_FROZEN
    period_start: str = "2018-07"
    period_end: str = "2025-06"
    cells: tuple[int, ...] = (1, 2, 3, 4)
    write_supabase: bool = True
    allow_overwrite_historical: bool = False
    resume_from: str | None = None
    workflow_id: str | None = None
