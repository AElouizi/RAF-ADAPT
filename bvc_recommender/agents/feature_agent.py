"""Feature Agent — calcule/vérifie features ; n'écrit jamais le livrable FINAL/mart."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

from bvc_recommender.agents.base import BaseAgent
from bvc_recommender.agents.contracts import AgentName, AgentResult, JobStatus, WorkflowMode, WorkflowRequest
from bvc_recommender.agents.scientific_guard import (
    AGENT_RUNS_DIR,
    FINAL_DIR,
    MART_DIR,
    feature_local_paths,
    refuse_recompute_frozen,
    sha256_file,
)
from bvc_recommender.config import (
    PROJECT_ROOT,
    TABLE_FEATURES_FONDAMENTALES,
    TABLE_FEATURES_INDICES,
    TABLE_FEATURES_TECHNIQUES,
)
from bvc_recommender.data.loader import fetch_table, get_supabase_client


class FeatureAgent(BaseAgent):
    name = AgentName.FEATURE

    def execute(self, job_id: str, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        if request.mode == WorkflowMode.COMPUTE_NEW:
            refuse_recompute_frozen(
                mode=request.mode.value,
                allow_overwrite=request.allow_overwrite_historical,
                period_start=request.period_start,
                period_end=request.period_end,
            )
            return self._compute(workflow_id, request)
        return self._ingest_frozen(workflow_id, request)

    def _ingest_frozen(self, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        local = feature_local_paths()
        supabase_counts: dict[str, int | str] = {}
        if request.write_supabase:
            try:
                client = get_supabase_client()
                for table in (
                    TABLE_FEATURES_TECHNIQUES,
                    TABLE_FEATURES_FONDAMENTALES,
                    TABLE_FEATURES_INDICES,
                ):
                    try:
                        df = fetch_table(client, table)
                        supabase_counts[table] = int(len(df))
                    except Exception as exc:
                        supabase_counts[table] = f"unavailable:{exc}"
            except Exception as exc:
                supabase_counts["client"] = f"unavailable:{exc}"
        else:
            supabase_counts["skipped"] = "write_supabase=False"

        frozen_ok = (
            Path(FINAL_DIR / "04_final_portfolios" / "C2_holdings_knee.csv").is_file()
            and Path(MART_DIR / "kpi_C1C4.csv").is_file()
        )
        if not local and not any(isinstance(v, int) and v > 0 for v in supabase_counts.values()):
            if not frozen_ok:
                return AgentResult(
                    status=JobStatus.FAILED,
                    error_message="Aucune feature locale, ni Supabase, ni livrable figé. L'administrateur doit alimenter la base.",
                )
            # Features déjà consommées par le livrable 2018-07→2025-06 : pas de recalcul.
        checksums = {k: sha256_file(p) for k, p in local.items()}
        out_dir = AGENT_RUNS_DIR / workflow_id / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = out_dir / "feature_manifest.txt"
        manifest.write_text(
            "\n".join(f"{k}\t{v}\t{local[k]}" for k, v in checksums.items()),
            encoding="utf-8",
        )
        return AgentResult(
            status=JobStatus.SUCCESS,
            event_type="FEATURES_READY",
            artifacts={"manifest": str(manifest)},
            metadata={
                "local_files": {k: str(v) for k, v in local.items()},
                "supabase": supabase_counts,
                "frozen_livrable_ok": frozen_ok,
            },
            rows_written=sum(v for v in supabase_counts.values() if isinstance(v, int)),
        )

    def _compute(self, workflow_id: str, request: WorkflowRequest) -> AgentResult:
        steps = [
            [sys.executable, "-m", "bvc_recommender.scripts.run_step2", "--use-processed"],
            [
                sys.executable,
                "-m",
                "bvc_recommender.scripts.run_step3",
                "--use-processed",
                "--use-features",
            ],
            [
                sys.executable,
                "-m",
                "bvc_recommender.scripts.run_step4",
                "--use-processed",
                "--start-year",
                "2010",
                "--end-year",
                "2026",
            ],
        ]
        for cmd in steps:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return AgentResult(
                    status=JobStatus.FAILED,
                    error_message=f"Feature compute échec {' '.join(cmd)} : {proc.stderr[-2000:]}",
                )
        return self._ingest_frozen(workflow_id, request)
