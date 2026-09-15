"""Configuration centrale — Étape 1 (chargement et préparation des données)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

# Historique cours / MASI : périmètre minimal
COURS_MIN_DATE = (os.getenv("COURS_MIN_DATE") or "2010-01-01").strip()
MASI_MIN_DATE = (os.getenv("MASI_MIN_DATE") or "2010-01-01").strip()

# Features fondamentales / techniques : pas de calcul avant cette date (3 juin 2015)
INDICATORS_MIN_DATE = (os.getenv("INDICATORS_MIN_DATE") or "2015-06-03").strip()

# Règle look-ahead (publication des résultats financiers BVC)
# S1 (clôture juin N)  → disponible fin septembre N
# S2 (clôture décembre N) → disponible fin avril N+1
PUBLICATION_MONTH_S1 = 9
PUBLICATION_MONTH_S2 = 4


def _table_name(env_key: str, default: str) -> str:
    return (os.getenv(env_key) or default).strip()


TABLE_FONDAMENTAUX_RS = _table_name("TABLE_FONDAMENTAUX_RS", "fondamentaux_rs")
# Table large de référence pour les ratios fondamentaux (Étape 2)
TABLE_FONDAMENTAUX_RS_BIS = _table_name("TABLE_FONDAMENTAUX_RS_BIS", "fondamentaux_rs_bis")
TABLE_DONNEES_FINANCIERS = _table_name("TABLE_DONNEES_FINANCIERS", "donnees_financieres")
TABLE_COURS_HISTORIQUE = _table_name(
    "TABLE_COURS_HISTORIQUE", "market_data_cours_historique"
)
TABLE_INDICES_HISTORIQUE = _table_name(
    "TABLE_INDICES_HISTORIQUE", "market_data_indices_historique"
)
TABLE_FEATURES_FONDAMENTALES = _table_name(
    "TABLE_FEATURES_FONDAMENTALES", "features_fondamentales"
)
TABLE_FEATURES_TECHNIQUES = _table_name(
    "TABLE_FEATURES_TECHNIQUES", "features_techniques"
)
TABLE_FEATURES_INDICES = _table_name("TABLE_FEATURES_INDICES", "features_indices")
TABLE_HISTO_CONST_IND = _table_name("TABLE_HISTO_CONST_IND", "histo_const_ind")
TABLE_INDICATEURS_FINANCIERS = _table_name(
    "TABLE_INDICATEURS_FINANCIERS", "indicateurs_financiers"
)
TABLE_AGENT_JOBS = _table_name("TABLE_AGENT_JOBS", "agent_jobs")
TABLE_AGENT_LOGS = _table_name("TABLE_AGENT_LOGS", "agent_logs")
TABLE_AGENT_EVENTS = _table_name("TABLE_AGENT_EVENTS", "agent_events")
TABLE_AGENT_RESULT_REGISTRY = _table_name(
    "TABLE_AGENT_RESULT_REGISTRY", "agent_result_registry"
)
TABLE_AGENT_RECOMMENDATIONS = _table_name(
    "TABLE_AGENT_RECOMMENDATIONS", "agent_recommendations"
)
TABLE_AGENT_PORTFOLIO_HOLDINGS = _table_name(
    "TABLE_AGENT_PORTFOLIO_HOLDINGS", "agent_portfolio_holdings"
)
TABLE_AGENT_BACKTEST_MONTHLY = _table_name(
    "TABLE_AGENT_BACKTEST_MONTHLY", "agent_backtest_monthly"
)
TABLE_AGENT_EVALUATION_KPIS = _table_name(
    "TABLE_AGENT_EVALUATION_KPIS", "agent_evaluation_kpis"
)

# Fallback CSV (export Supabase) si RLS bloque l'API anon
HISTO_CONST_IND_CSV = PROJECT_ROOT.parent / "histo_const_ind_final.csv"
HISTO_CONST_IND_CSV_LEGACY = PROJECT_ROOT.parent / "histo_const_ind_rows.csv"
INDICATEURS_FINANCIERS_CSV = PROJECT_ROOT.parent / "indicateurs_financiers_rows.csv"

REPORTS_DIR = PROJECT_ROOT / "bvc_recommender" / "reports"
RECOMMENDATIONS_DIR = REPORTS_DIR / "recommendations"
EXPORTS_DIR = PROJECT_ROOT / "bvc_recommender" / "exports"
DATA_PROCESSED_DIR = PROJECT_ROOT / "bvc_recommender" / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "bvc_recommender" / "data" / "features"
DATASET_DIR = PROJECT_ROOT / "bvc_recommender" / "data" / "datasets"

# Rebalancement portefeuille : monthly | quarterly
REBALANCE_FREQUENCY = (os.getenv("REBALANCE_FREQUENCY") or "monthly").strip().lower()
INDICATORS_EXCEL_PATH = EXPORTS_DIR / "indicateurs_bvc.xlsx"

# Dataset ML — horizon cible et splits temporels (brief projet)
FORWARD_HORIZON_DAYS = int(os.getenv("FORWARD_HORIZON_DAYS") or "63")
SPLIT_TRAIN_END = (os.getenv("SPLIT_TRAIN_END") or "2020-12-31").strip()
SPLIT_VAL_END = (os.getenv("SPLIT_VAL_END") or "2022-12-31").strip()
SPLIT_TEST_START = (os.getenv("SPLIT_TEST_START") or "2023-01-01").strip()

# Étape 4 — détection régime HMM (fit ≤ SPLIT_TRAIN_END)
REGIME_START_YEAR = int(os.getenv("REGIME_START_YEAR") or "2015")
REGIME_END_YEAR = int(os.getenv("REGIME_END_YEAR") or "2025")

# Liquidité — seuil VMQ (MAD)
LIQUIDITY_VMQ_THRESHOLD_MAD = 500_000

# Secteurs financiers (EV/EBITDA et FCF Yield = NA)
FINANCIAL_SECTOR_KEYWORDS = ("banque", "assurance", "financement", "leasing")


def load_env_file(path: Path | None = None) -> None:
    """Charge le fichier .env sans écraser les variables déjà définies."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off", "n"}


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    verify_ssl: bool = True
    service_role_key: str = ""
    database_url: str = ""

    def validate(self) -> None:
        missing = []
        if not self.supabase_url or "YOUR_PROJECT" in self.supabase_url.upper():
            missing.append("SUPABASE_URL")
        if not self.supabase_key or self.supabase_key.startswith("your-"):
            missing.append("SUPABASE_KEY / SUPABASE_ANON_KEY")
        if missing:
            raise ValueError(
                f"Variables manquantes ou placeholders : {', '.join(missing)}. "
                "Projet attendu : Analyste_IA_26 (pungyebagtntbocoewir). "
                "Dashboard → Settings → API → URL + clé anon."
            )
        url_ref = self._url_project_ref()
        key_ref = _jwt_project_ref(self.supabase_key)
        if url_ref and key_ref and url_ref != key_ref:
            raise ValueError(
                f"URL projet ({url_ref}) et clé anon (ref JWT {key_ref}) ne correspondent pas. "
                "Utilisez les deux depuis le même projet Supabase (Analyste_IA_26)."
            )

    def _url_project_ref(self) -> str | None:
        url = self.supabase_url.rstrip("/")
        if ".supabase.co" not in url:
            return None
        host = url.split("//", 1)[-1]
        return host.split(".")[0] or None


def _jwt_project_ref(token: str) -> str | None:
    try:
        import base64
        import json

        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        ref = data.get("ref")
        return str(ref) if ref else None
    except Exception:
        return None


@lru_cache
def get_settings() -> Settings:
    load_env_file()
    return Settings(
        supabase_url=(
            os.getenv("SUPABASE_URL")
            or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            or ""
        ).strip(),
        supabase_key=(
            os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_KEY")
            or ""
        ).strip(),
        verify_ssl=_env_bool("SUPABASE_VERIFY_SSL", default=True),
        service_role_key=(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip(),
        database_url=(os.getenv("DATABASE_URL") or "").strip(),
    )
