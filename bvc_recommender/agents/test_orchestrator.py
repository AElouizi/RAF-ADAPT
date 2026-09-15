"""Tests d'orchestration : pas de recalcul, pas d'overwrite du livrable figé."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.agents.contracts import AgentResult, JobStatus, WorkflowMode, WorkflowRequest
from bvc_recommender.agents.feature_agent import FeatureAgent
from bvc_recommender.agents.orchestrator import run_workflow
from bvc_recommender.agents.scientific_guard import (
    FROZEN_END,
    FROZEN_START,
    frozen_artifact_manifest,
    refuse_recompute_frozen,
)


def test_frozen_files_untouched_after_ingest() -> None:
    before = frozen_artifact_manifest()
    assert before, "Livrable figé manquant (kpi / holdings / mart)."
    out = run_workflow(
        WorkflowRequest(
            mode=WorkflowMode.INGEST_FROZEN,
            period_start=FROZEN_START,
            period_end=FROZEN_END,
            cells=(1, 2, 3, 4),
            write_supabase=False,
        )
    )
    after = frozen_artifact_manifest()
    assert out["status"] == JobStatus.SUCCESS.value, out
    assert before == after
    ev = out["results"]["evaluation"]
    assert ev["status"] == JobStatus.SUCCESS.value
    assert ev["rows"] >= 5


def test_failed_feature_blocks_pipeline() -> None:
    def _fail(self, job_id, workflow_id, request):
        return AgentResult(status=JobStatus.FAILED, error_message="boom-test")

    with patch.object(FeatureAgent, "execute", _fail):
        out = run_workflow(
            WorkflowRequest(
                mode=WorkflowMode.INGEST_FROZEN,
                period_start=FROZEN_START,
                period_end=FROZEN_END,
                cells=(2,),
                write_supabase=False,
            )
        )
    assert out["status"] == JobStatus.FAILED.value
    assert out["failed_agent"] == "feature"
    assert "selection" not in out["results"]


def test_compute_new_refused_on_frozen_window() -> None:
    raised = False
    try:
        refuse_recompute_frozen(
            mode="compute_new",
            allow_overwrite=False,
            period_start="2018-07",
            period_end="2025-06",
        )
    except PermissionError:
        raised = True
    assert raised


if __name__ == "__main__":
    test_compute_new_refused_on_frozen_window()
    print("ok refuse")
    test_failed_feature_blocks_pipeline()
    print("ok fail-stop")
    test_frozen_files_untouched_after_ingest()
    print("ok ingest")
    print("ALL TESTS PASSED")
