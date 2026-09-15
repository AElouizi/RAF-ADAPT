# -*- coding: utf-8 -*-
"""Graphique d'évolution de performance — tous portefeuilles NSGA vs MASI.

Source fiable : fact_performance_daily.csv (richesse base 100, OOS).
Les séries mensuelles c1c5/fact_portfolio_monthly sont volontairement
ignorées (anomalie de rendements aux frontières de folds).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MART = Path(__file__).resolve().parent / "outputs" / "platform_mart"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "reports" / "figures"
LIV = ROOT / "livrables" / "figures_final"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LIV.mkdir(parents=True, exist_ok=True)

daily = pd.read_csv(MART / "fact_performance_daily.csv", parse_dates=["date"]).sort_values(
    "date"
)

# C3 = champion / portefeuille optimal NSGA (P_équilibre) selon meta platform
series_daily = [
    ("C1", "C1 Ridge", "#1f77b4", 1.4, "-"),
    ("C2", "C2 Random Forest", "#b45309", 1.4, "-"),
    ("C3", "C3 LightGBM — optimal NSGA", "#e85d04", 2.6, "-"),
    ("C4", "C4 Hybrid", "#15803d", 1.4, "-"),
    ("EqualWeight", "Equal Weight", "#64748b", 1.3, "--"),
    ("MASI", "MASI", "#1d4ed8", 2.2, "-"),
]

fig, ax = plt.subplots(figsize=(12.5, 6.2))
for col, lab, color, lw, ls in series_daily:
    ax.plot(daily["date"], daily[col], label=lab, color=color, lw=lw, ls=ls)
ax.axhline(100, color="#94a3b8", lw=0.9, ls=":", zorder=0)
ax.set_title(
    "Évolution de la performance — portefeuilles NSGA (P_équilibre) vs MASI\n"
    "Historique OOS complet · base 100 · août 2018 → juillet 2025"
)
ax.set_ylabel("Richesse cumulée (base 100)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", frameon=True, fontsize=9)
ax.grid(True, alpha=0.28)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
last = daily.iloc[-1]
for col, _lab, color, _lw, _ls in series_daily:
    ax.annotate(
        f"{last[col]:.0f}",
        xy=(last["date"], last[col]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
    )
fig.tight_layout()
out1 = OUT_DIR / "evolution_performance_tous_portefeuilles.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
fig.savefig(LIV / out1.name, dpi=200, bbox_inches="tight")
plt.close()
print("saved", out1)

# Focus optimal NSGA vs MASI vs EW
fig, ax = plt.subplots(figsize=(12.5, 6.0))
focus = [
    ("C3", "Portefeuille optimal NSGA (C3 / P_équilibre)", "#e85d04", 2.6, "-"),
    ("EqualWeight", "Equal Weight", "#64748b", 1.5, "--"),
    ("MASI", "MASI", "#1d4ed8", 2.2, "-"),
]
for col, lab, color, lw, ls in focus:
    ax.plot(daily["date"], daily[col], label=lab, color=color, lw=lw, ls=ls)
ax.axhline(100, color="#94a3b8", lw=0.9, ls=":", zorder=0)
ax.set_title(
    "Portefeuille optimal NSGA vs MASI — richesse cumulée (base 100)\n"
    "Août 2018 → juillet 2025"
)
ax.set_ylabel("Richesse cumulée (base 100)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.grid(True, alpha=0.28)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
out2 = OUT_DIR / "evolution_performance_optimal_NSGA_vs_MASI.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
fig.savefig(LIV / out2.name, dpi=200, bbox_inches="tight")
plt.close()
print("saved", out2)

final = {lab: round(float(last[col]), 2) for col, lab, *_ in series_daily}
print("FINAL", final)

# Canvas payload: fin de mois
monthly = daily.set_index("date")[[c for c, *_ in series_daily]].resample("ME").last()
monthly.index = monthly.index.strftime("%Y-%m")
monthly = monthly.round(2)
# un point sur deux pour alléger
export_c = monthly.iloc[::2]
rename = {c: lab for c, lab, *_ in series_daily}
export_c = export_c.rename(columns=rename)
payload = {
    "categories": export_c.index.tolist(),
    "series": {c: export_c[c].tolist() for c in export_c.columns},
    "final": final,
    "start": str(daily["date"].min().date()),
    "end": str(daily["date"].max().date()),
}
(OUT_DIR / "evolution_performance_canvas_data.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
)
print("canvas_points", len(export_c))
