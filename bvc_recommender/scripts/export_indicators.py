"""
Export des indicateurs vers Excel + Supabase.

Usage :
    py -m bvc_recommender.scripts.export_indicators
    py -m bvc_recommender.scripts.export_indicators --excel-only
    py -m bvc_recommender.scripts.export_indicators --supabase-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import INDICATORS_EXCEL_PATH, REPORTS_DIR, load_env_file  # noqa: E402
from bvc_recommender.data.loader import get_supabase_client  # noqa: E402
from bvc_recommender.features.exporter import export_and_publish  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export indicateurs BVC")
    parser.add_argument("--excel-only", action="store_true")
    parser.add_argument("--supabase-only", action="store_true")
    parser.add_argument("--excel-path", default=str(INDICATORS_EXCEL_PATH))
    args = parser.parse_args()

    do_excel = not args.supabase_only
    do_supabase = not args.excel_only

    client = None
    if do_supabase:
        try:
            client = get_supabase_client()
            logger.info("Client Supabase connecté.")
        except Exception as exc:
            logger.warning("Supabase indisponible : %s — export local uniquement.", exc)

    result = export_and_publish(
        client,
        excel=do_excel,
        supabase=do_supabase,
        excel_path=Path(args.excel_path),
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Export indicateurs",
        **result,
    }
    report_path = REPORTS_DIR / "indicators_export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if result.get("excel_path"):
        logger.info("Excel : %s", result["excel_path"])
    if result.get("supabase"):
        for key, status in result["supabase"].items():
            if isinstance(status, dict):
                ok = status.get("ok", False)
                rows = status.get("rows_written", 0)
                err = status.get("error")
                if ok:
                    logger.info("Supabase %s : %s lignes", key, rows)
                else:
                    logger.warning("Supabase %s échoué : %s", key, err)

    logger.info("Rapport export : %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
