"""Rapport de qualité des données — Étape 1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bvc_recommender.data.masi_coverage import check_masi_coverage


@dataclass
class TableQuality:
    name: str
    row_count: int
    column_count: int
    date_min: str | None = None
    date_max: str | None = None
    missing_pct: dict[str, float] = field(default_factory=dict)
    status: str = "OK"
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    generated_at: str
    project: str = "BVC Recommender — Étape 1"
    tables: list[TableQuality] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Rapport qualité des données — Étape 1",
            "",
            f"- **Projet** : {self.project}",
            f"- **Généré le** : {self.generated_at}",
            "",
        ]

        if self.blockers:
            lines.extend(["## Bloquants", ""])
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")

        if self.warnings:
            lines.extend(["## Avertissements", ""])
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.extend(
            [
                "## Synthèse par table",
                "",
                "| Table | Lignes | Colonnes | Date min | Date max | Statut |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        for t in self.tables:
            lines.append(
                f"| {t.name} | {t.row_count:,} | {t.column_count} | "
                f"{t.date_min or '—'} | {t.date_max or '—'} | {t.status} |"
            )

        lines.extend(["", "## Valeurs manquantes (% par colonne)", ""])
        for t in self.tables:
            if not t.missing_pct:
                continue
            lines.append(f"### {t.name}")
            lines.append("")
            top = sorted(t.missing_pct.items(), key=lambda x: -x[1])[:15]
            for col, pct in top:
                lines.append(f"- `{col}` : {pct:.1f} %")
            lines.append("")

        if self.metrics:
            lines.extend(["## Métriques globales", ""])
            for key, val in self.metrics.items():
                lines.append(f"- **{key}** : {val}")
            lines.append("")

        masi = self.metrics.get("masi_coverage")
        if isinstance(masi, dict):
            lines.extend(
                [
                    "## Vérification MASI (benchmark)",
                    "",
                    f"- **code_index** : {masi.get('code_index', 'MASI')}",
                    f"- **Lignes MASI** : {masi.get('row_count', 0):,}",
                    f"- **Date min** : {masi.get('date_min') or '—'}",
                    f"- **Date max** : {masi.get('date_max') or '—'}",
                    f"- **Couverture depuis 2010** : "
                    f"{'Oui' if masi.get('covers_from_2010') else 'Non'}",
                    f"- **Statut** : {masi.get('status', '—')}",
                ]
            )
            for note in masi.get("notes") or []:
                lines.append(f"- **Note** : {note}")
            lines.append("")

        lines.extend(
            [
                "## Règles appliquées (Étape 1)",
                "",
                "- **Look-ahead** : S1 (juin N) → disponible fin septembre N ; "
                "S2 (décembre N) → disponible fin avril N+1 (`publication_date`)",
                "- **Valeurs manquantes** : carry-forward (ffill) uniquement, pas de bfill",
                "- **Données financières** : filtre `statut_validation=VALIDE`, "
                "comptes `CONSOLIDE` pour donnees_financieres",
                "- **Cours** : périmètre depuis `COURS_MIN_DATE` (défaut 2010-01-01)",
                "- **random_state** : 42 (reproductibilité)",
                "",
            ]
        )
        return "\n".join(lines)


def _detect_date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    date_candidates = [
        "date_fin",
        "date",
        "date_cours",
        "date_index",
        "publication_date",
    ]
    for col in date_candidates:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce").dropna()
            if not s.empty:
                return str(s.min().date()), str(s.max().date())
    return None, None


def _missing_pct(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    return {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns}


def assess_table(name: str, df: pd.DataFrame) -> TableQuality:
    notes: list[str] = []
    status = "OK"
    if df.empty:
        status = "BLOCKER" if "market_data" in name else "WARNING"
        notes.append("table vide")
        return TableQuality(name, 0, 0, status=status, notes=notes)

    dmin, dmax = _detect_date_range(df)
    missing = _missing_pct(df)
    median_missing = float(pd.Series(list(missing.values())).median()) if missing else 0.0
    if median_missing > 50:
        status = "WARNING"
        notes.append(f"missing médian {median_missing:.1f}%")

    return TableQuality(
        name=name,
        row_count=len(df),
        column_count=len(df.columns),
        date_min=dmin,
        date_max=dmax,
        missing_pct=missing,
        status=status,
        notes=notes,
    )


def build_quality_report(
    raw: dict[str, pd.DataFrame],
    cleaned: dict[str, pd.DataFrame],
    merged_summary: dict[str, Any] | None = None,
) -> QualityReport:
    report = QualityReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    for name, df in raw.items():
        report.tables.append(assess_table(f"raw/{name}", df))

    for name, df in cleaned.items():
        report.tables.append(assess_table(f"clean/{name}", df))

    if merged_summary:
        report.metrics.update(merged_summary)

    cours = cleaned.get("market_data_cours_historique", pd.DataFrame())
    idx = cleaned.get("market_data_indices_historique", pd.DataFrame())
    fond = cleaned.get("fondamentaux_rs", pd.DataFrame())
    fin = cleaned.get("donnees_financieres", pd.DataFrame())

    if cours.empty:
        report.blockers.append(
            "market_data_cours_historique vide — features techniques et backtest impossibles"
        )
    if idx.empty:
        report.blockers.append(
            "market_data_indices_historique vide — benchmark MASI indisponible"
        )
    if fond.empty and fin.empty:
        report.blockers.append("Aucune source fondamentale disponible")
    elif fond.empty:
        report.warnings.append(
            "fondamentaux_rs vide — vérifier RLS / nom de table (fondamentaux_rs vs fondamentaux_RS)"
        )
    if fin.empty:
        report.blockers.append("donnees_financieres vide")

    if not fond.empty and "publication_date" in fond.columns:
        report.metrics["publication_date_appliquee"] = "fondamentaux_rs (S1/S2)"

    masi_check = check_masi_coverage(idx)
    report.metrics["masi_coverage"] = masi_check
    report.metrics["masi_covers_from_2010"] = masi_check["covers_from_2010"]
    if masi_check["status"] == "WARNING":
        report.warnings.extend(masi_check["notes"])
    elif masi_check["status"] == "BLOCKER":
        report.blockers.extend(masi_check["notes"])

    return report


def write_quality_report(
    report: QualityReport,
    output_dir: Path,
    *,
    basename: str = "data_quality_report",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{basename}.md"
    json_path = output_dir / f"{basename}.json"

    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": report.generated_at,
                "project": report.project,
                "tables": [asdict(t) for t in report.tables],
                "metrics": report.metrics,
                "blockers": report.blockers,
                "warnings": report.warnings,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md_path, json_path
