"""CLI orchestrateur.

  py -m bvc_recommender.agents --mode ingest_frozen --cells 1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.agents.contracts import WorkflowMode, WorkflowRequest
from bvc_recommender.agents.orchestrator import retry_failed_step, run_workflow
from bvc_recommender.config import load_env_file

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description="Orchestrateur multi-agents (sans recalcul scientifique figé)")
    p.add_argument("--mode", choices=("ingest_frozen", "compute_new"), default="ingest_frozen")
    p.add_argument("--period-start", default="2018-07")
    p.add_argument("--period-end", default="2025-06")
    p.add_argument("--cells", default="1,2,3,4")
    p.add_argument("--resume-from", default=None, help="feature|selection|allocation|backtest|evaluation")
    p.add_argument("--workflow-id", default=None)
    p.add_argument("--no-supabase", action="store_true")
    args = p.parse_args()
    cells = tuple(int(x) for x in args.cells.split(",") if x.strip())
    req = WorkflowRequest(
        mode=WorkflowMode(args.mode),
        period_start=args.period_start,
        period_end=args.period_end,
        cells=cells,
        write_supabase=not args.no_supabase,
        resume_from=args.resume_from,
        workflow_id=args.workflow_id,
    )
    if args.resume_from and args.workflow_id:
        out = retry_failed_step(args.workflow_id, args.resume_from, req)
    else:
        out = run_workflow(req)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0 if out.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
