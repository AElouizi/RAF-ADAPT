"""Recharge uniquement market_data_indices_historique + met à jour processed/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATA_PROCESSED_DIR, load_env_file  # noqa: E402
from bvc_recommender.data.cleaner import clean_market_data_indices  # noqa: E402
from bvc_recommender.data.loader import get_supabase_client, load_indices_historique  # noqa: E402
from bvc_recommender.data.masi_coverage import check_masi_coverage  # noqa: E402

load_env_file(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    client = get_supabase_client()
    cache = DATA_PROCESSED_DIR / "raw_cache"
    raw = load_indices_historique(client, cache_dir=cache, refresh=True)
    check = check_masi_coverage(raw)
    logger.info(
        "MASI : %s lignes, %s -> %s, couverture 2010=%s",
        check["row_count"],
        check["date_min"],
        check["date_max"],
        check["covers_from_2010"],
    )
    cleaned = clean_market_data_indices(raw)
    out = DATA_PROCESSED_DIR / "market_data_indices_historique.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False)
    logger.info("Sauvegarde %s (%s lignes)", out, len(cleaned))
    return 0 if check["covers_from_2010"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
