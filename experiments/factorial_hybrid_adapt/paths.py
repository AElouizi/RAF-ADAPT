"""Chemins locaux de l'expérience (aucun chevauchement avec TFT / step5)."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
SELECTIONS_DIR = OUTPUT_DIR / "selections"


def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, SELECTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
