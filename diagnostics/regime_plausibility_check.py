"""
Diagnostic lecture seule — plausibilite economique du regime HMM vs MASI.

Livrables :
  - diagnostics/regime_plausibility_check.py (ce script)
  - diagnostics/regime_vs_masi.png
  - diagnostics/regime_plausibility_report.txt (resume)

Usage :
  py -3 diagnostics/regime_plausibility_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "regime_vs_masi.png"
OUT_REPORT = OUT_DIR / "regime_plausibility_report.txt"
OUT_CSV = OUT_DIR / "regime_plausibility_months.csv"

REGIME_COLORS = {"bear": "#d62728", "sideways": "#7f7f7f", "bull": "#2ca02c"}
NEAR_ZERO = 0.002  # |ret| < 0.2% = proche de zero


def _dominant(row: pd.Series) -> str:
    d = {
        "bull": int(row.get("is_bull", 0) or 0),
        "sideways": int(row.get("is_sideways", 0) or 0),
        "bear": int(row.get("is_bear", 0) or 0),
    }
    return max(d, key=d.get)


def _load_regime_history() -> pd.DataFrame:
    path = ROOT / "bvc_recommender" / "data" / "datasets" / "regime_history.parquet"
    h = pd.read_parquet(path).sort_values(["year", "month"]).reset_index(drop=True)
    h["regime"] = h.apply(_dominant, axis=1)
    return h


def _load_masi_eom() -> pd.DataFrame:
    """Rendements mensuels MASI depuis la serie indice deja en base (EOM)."""
    path = (
        ROOT
        / "bvc_recommender"
        / "data"
        / "processed"
        / "market_data_indices_historique.parquet"
    )
    idx = pd.read_parquet(path)
    masi = idx[idx["code_index"].astype(str).str.upper() == "MASI"].copy()
    masi["date_index"] = pd.to_datetime(masi["date_index"], errors="coerce")
    masi["valeur_index"] = pd.to_numeric(masi["valeur_index"], errors="coerce")
    masi = masi.dropna(subset=["date_index", "valeur_index"]).sort_values("date_index")
    masi = masi.drop_duplicates("date_index", keep="last")
    masi["year"] = masi["date_index"].dt.year
    masi["month"] = masi["date_index"].dt.month

    eom = (
        masi.groupby(["year", "month"], as_index=False)
        .tail(1)[["year", "month", "date_index", "valeur_index"]]
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    eom["masi_ret_1m"] = eom["valeur_index"].pct_change()
    prev = eom["masi_ret_1m"]
    eom["masi_ret_3m_prev"] = (
        (1 + prev.shift(3)) * (1 + prev.shift(2)) * (1 + prev.shift(1)) - 1
    )
    eom["masi_ret_3m_fwd"] = (
        (1 + prev.shift(-1)) * (1 + prev.shift(-2)) * (1 + prev.shift(-3)) - 1
    )
    return eom


def _print_month_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} (n={len(df)}) ===")
    hdr = f"{'mois':<10} {'regime':<10} {'ret_1m':>10} {'ret_3m_prev':>12} {'ret_3m_fwd':>12}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        def fmt(x: float) -> str:
            return "n/a" if pd.isna(x) else f"{100 * x:+.2f}%"

        print(
            f"{r['month_label']:<10} {r['regime']:<10} "
            f"{fmt(r['masi_ret_1m']):>10} {fmt(r['masi_ret_3m_prev']):>12} "
            f"{fmt(r['masi_ret_3m_fwd']):>12}"
        )


def _regime_stats(df: pd.DataFrame, regime: str) -> dict:
    sub = df[df["regime"] == regime].dropna(subset=["masi_ret_1m"])
    out: dict = {"regime": regime, "n": int(len(sub))}
    for col, key in [
        ("masi_ret_1m", "ret_1m"),
        ("masi_ret_3m_fwd", "ret_3m_fwd"),
        ("masi_ret_3m_prev", "ret_3m_prev"),
    ]:
        s = sub[col].dropna()
        out[f"{key}_mean"] = float(s.mean()) if len(s) else float("nan")
        out[f"{key}_median"] = float(s.median()) if len(s) else float("nan")
    if regime == "bear":
        out["pct_neg_1m"] = 100.0 * float((sub["masi_ret_1m"] < 0).mean()) if len(sub) else float("nan")
        out["pct_pos_1m"] = 100.0 * float((sub["masi_ret_1m"] > 0).mean()) if len(sub) else float("nan")
    elif regime == "bull":
        out["pct_pos_1m"] = 100.0 * float((sub["masi_ret_1m"] > 0).mean()) if len(sub) else float("nan")
        out["pct_neg_1m"] = 100.0 * float((sub["masi_ret_1m"] < 0).mean()) if len(sub) else float("nan")
    else:
        out["pct_neg_1m"] = 100.0 * float((sub["masi_ret_1m"] < 0).mean()) if len(sub) else float("nan")
        out["pct_pos_1m"] = 100.0 * float((sub["masi_ret_1m"] > 0).mean()) if len(sub) else float("nan")
    return out


def _print_stats(st: dict) -> None:
    print(f"\n--- Stats {st['regime']} (n={st['n']}) ---")
    print(
        f"  ret_1m     mean={100*st['ret_1m_mean']:+.2f}%  "
        f"median={100*st['ret_1m_median']:+.2f}%"
    )
    print(
        f"  ret_3m_fwd mean={100*st['ret_3m_fwd_mean']:+.2f}%  "
        f"median={100*st['ret_3m_fwd_median']:+.2f}%"
    )
    print(
        f"  ret_3m_prev mean={100*st['ret_3m_prev_mean']:+.2f}%  "
        f"median={100*st['ret_3m_prev_median']:+.2f}%"
    )
    if st["regime"] == "bear":
        print(f"  % mois avec ret_1m < 0 : {st['pct_neg_1m']:.1f}%")
    elif st["regime"] == "bull":
        print(f"  % mois avec ret_1m > 0 : {st['pct_pos_1m']:.1f}%")


def _suspects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bear = df[df["regime"] == "bear"].dropna(subset=["masi_ret_1m"])
    bull = df[df["regime"] == "bull"].dropna(subset=["masi_ret_1m"])
    # Bear incoherent : ret positif ou proche de zero
    bear_bad = bear[(bear["masi_ret_1m"] > -NEAR_ZERO)].copy()
    # Bull incoherent : ret negatif
    bull_bad = bull[bull["masi_ret_1m"] < 0].copy()
    return bear_bad, bull_bad


def _episodes(regimes: list[str], target: str) -> list[int]:
    """Longueurs des sequences consecutives du regime cible."""
    lengths: list[int] = []
    cur = 0
    for r in regimes:
        if r == target:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def _plot_masi_vs_regime(df: pd.DataFrame, eom_full: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    # Courbe MASI journaliere sur 2015-2025
    path = (
        ROOT
        / "bvc_recommender"
        / "data"
        / "processed"
        / "market_data_indices_historique.parquet"
    )
    idx = pd.read_parquet(path)
    masi = idx[idx["code_index"].astype(str).str.upper() == "MASI"].copy()
    masi["date_index"] = pd.to_datetime(masi["date_index"], errors="coerce")
    masi["valeur_index"] = pd.to_numeric(masi["valeur_index"], errors="coerce")
    masi = masi.dropna(subset=["date_index", "valeur_index"]).sort_values("date_index")
    masi = masi[
        (masi["date_index"].dt.year >= 2015) & (masi["date_index"].dt.year <= 2025)
    ]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        masi["date_index"],
        masi["valeur_index"],
        color="#1f77b4",
        linewidth=1.2,
        label="MASI",
        zorder=3,
    )

    # Bandes mensuelles
    for _, row in df.iterrows():
        start = pd.Timestamp(year=int(row["year"]), month=int(row["month"]), day=1)
        if int(row["month"]) == 12:
            end = pd.Timestamp(year=int(row["year"]) + 1, month=1, day=1)
        else:
            end = pd.Timestamp(year=int(row["year"]), month=int(row["month"]) + 1, day=1)
        ax.axvspan(
            start,
            end,
            facecolor=REGIME_COLORS[row["regime"]],
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )

    ax.set_title("MASI vs régime HMM détecté (2015–2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Niveau MASI")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, zorder=0)
    patches = [
        mpatches.Patch(color=REGIME_COLORS["bear"], alpha=0.4, label="Bear"),
        mpatches.Patch(color=REGIME_COLORS["sideways"], alpha=0.4, label="Sideways"),
        mpatches.Patch(color=REGIME_COLORS["bull"], alpha=0.4, label="Bull"),
    ]
    ax.legend(handles=patches + [ax.lines[0]], loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure ecrite : {out_path}")


def main() -> int:
    lines: list[str] = []

    def both(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    both("=" * 72)
    both("DIAGNOSTIC PLAUSIBILITE REGIME HMM vs MASI")
    both("=" * 72)

    h = _load_regime_history()
    eom = _load_masi_eom()
    df = h.merge(
        eom[
            [
                "year",
                "month",
                "date_index",
                "valeur_index",
                "masi_ret_1m",
                "masi_ret_3m_prev",
                "masi_ret_3m_fwd",
            ]
        ],
        on=["year", "month"],
        how="left",
    )
    df.to_csv(OUT_CSV, index=False, float_format="%.8f")
    both(f"Mois charges : {len(df)} | export {OUT_CSV.name}")
    both(f"Distribution : {df['regime'].value_counts().to_dict()}")

    # 1) Tables par regime
    bear = df[df["regime"] == "bear"]
    bull = df[df["regime"] == "bull"]
    side = df[df["regime"] == "sideways"]
    # echantillon sideways : 1 tous les ~5 mois pour lisibilite
    side_sample = side.iloc[:: max(1, len(side) // 12)].head(12)

    _print_month_table(bear, "MOIS BEAR")
    _print_month_table(bull, "MOIS BULL")
    _print_month_table(side_sample, "ECHANTILLON SIDEWAYS")

    # 2) Stats agregees
    both("\n" + "=" * 72)
    both("2. STATISTIQUES AGREGEES PAR REGIME")
    both("=" * 72)
    stats = {r: _regime_stats(df, r) for r in ("bear", "sideways", "bull")}
    for r in ("bear", "sideways", "bull"):
        _print_stats(stats[r])
        st = stats[r]
        lines.append(
            f"{r}: n={st['n']} ret_1m_mean={st['ret_1m_mean']:.4f} "
            f"ret_3m_fwd_mean={st['ret_3m_fwd_mean']:.4f} "
            f"pct_neg={st.get('pct_neg_1m')} pct_pos={st.get('pct_pos_1m')}"
        )

    # 3) Cas suspects
    both("\n" + "=" * 72)
    both("3. CAS SUSPECTS (incoherence regime vs ret_1m MASI)")
    both("=" * 72)
    bear_bad, bull_bad = _suspects(df)
    both(
        f"\nBear avec ret_1m >= -{100*NEAR_ZERO:.1f}% (positif ou ~0) : "
        f"{len(bear_bad)}/{int((df['regime']=='bear').sum())}"
    )
    if len(bear_bad):
        for _, r in bear_bad.sort_values("masi_ret_1m", ascending=False).iterrows():
            r3p = (
                f"{100 * r['masi_ret_3m_prev']:+.2f}%"
                if pd.notna(r["masi_ret_3m_prev"])
                else "n/a"
            )
            r3f = (
                f"{100 * r['masi_ret_3m_fwd']:+.2f}%"
                if pd.notna(r["masi_ret_3m_fwd"])
                else "n/a"
            )
            both(
                f"  - {r['month_label']}: ret_1m={100 * r['masi_ret_1m']:+.2f}%  "
                f"ret_3m_prev={r3p}  ret_3m_fwd={r3f}"
            )
    both(
        f"\nBull avec ret_1m < 0 : "
        f"{len(bull_bad)}/{int((df['regime']=='bull').sum())}"
    )
    if len(bull_bad):
        for _, r in bull_bad.iterrows():
            both(f"  - {r['month_label']}: ret_1m={100*r['masi_ret_1m']:+.2f}%")
    else:
        both("  (aucun)")

    # 4) Persistance
    both("\n" + "=" * 72)
    both("4. PERSISTANCE DES EPISODES")
    both("=" * 72)
    regimes = df["regime"].tolist()
    labels = df["month_label"].tolist()
    for target in ("bear", "bull", "sideways"):
        lengths = _episodes(regimes, target)
        if not lengths:
            both(f"  {target}: aucun episode")
            continue
        n_iso = sum(1 for L in lengths if L == 1)
        both(
            f"  {target}: {len(lengths)} episodes | "
            f"duree moyenne={np.mean(lengths):.2f} mois | "
            f"mediane={np.median(lengths):.1f} | "
            f"max={max(lengths)} | "
            f"isolés (1 mois)={n_iso}/{len(lengths)} ({100*n_iso/len(lengths):.0f}%)"
        )
        if target == "bear":
            # detail sequences
            both("  Sequences bear :")
            i = 0
            while i < len(regimes):
                if regimes[i] != "bear":
                    i += 1
                    continue
                j = i
                while j < len(regimes) and regimes[j] == "bear":
                    j += 1
                both(f"    {labels[i]} → {labels[j-1]}  ({j-i} mois)")
                i = j

    # 5) Plot
    both("\n" + "=" * 72)
    both("5. VISUALISATION")
    both("=" * 72)
    _plot_masi_vs_regime(df, eom, OUT_PNG)

    # 6) Verdict
    both("\n" + "=" * 72)
    both("6. RESUME / VERDICT")
    both("=" * 72)
    st_b = stats["bear"]
    st_u = stats["bull"]
    bear_lengths = _episodes(regimes, "bear")
    pct_iso = (
        100.0 * sum(1 for L in bear_lengths if L == 1) / len(bear_lengths)
        if bear_lengths
        else 100.0
    )
    mean_ep = float(np.mean(bear_lengths)) if bear_lengths else 0.0

    both(
        f"\n1) % mois bear avec ret_1m MASI negatif : "
        f"{st_b['pct_neg_1m']:.1f}% ({int(round(st_b['pct_neg_1m']/100*st_b['n']))}/{st_b['n']})"
    )
    both(
        f"   ret_1m moyen bear={100*st_b['ret_1m_mean']:+.2f}% | "
        f"ret_3m_fwd moyen bear={100*st_b['ret_3m_fwd_mean']:+.2f}%"
    )
    both(
        f"   (bull) % ret_1m positif : {st_u['pct_pos_1m']:.1f}% | "
        f"ret_1m moyen={100*st_u['ret_1m_mean']:+.2f}%"
    )

    both(
        f"\n2) Episodes bear : duree moyenne={mean_ep:.2f} mois, "
        f"isoles={pct_iso:.0f}% des episodes"
    )
    if mean_ep >= 2.0 and pct_iso <= 50:
        both("   → Sequences plutot groupees (coherent marche frontiere).")
        persist_ok = True
    else:
        both("   → Trop d'episodes isoles / courts (transitions erratiques).")
        persist_ok = False

    both(
        f"\n3) Cas suspects : {len(bear_bad)} bear incoherents, "
        f"{len(bull_bad)} bull incoherents"
    )
    if len(bear_bad):
        both("   Bear suspects : " + ", ".join(bear_bad["month_label"].tolist()))
    if len(bull_bad):
        both("   Bull suspects : " + ", ".join(bull_bad["month_label"].tolist()))

    # Criteres verdict
    # - >=60% bear avec ret negatif
    # - ret_1m moyen bear < 0
    # - persist_ok
    # - bull: si n>0, majorite ret positifs OU mean>0
    coherence_ok = (
        st_b["pct_neg_1m"] >= 60.0
        and st_b["ret_1m_mean"] < 0
        and (st_u["n"] == 0 or st_u["pct_pos_1m"] >= 50.0 or st_u["ret_1m_mean"] > 0)
        and persist_ok
    )
    # Tolérance : quelques mois bear positifs en fin d'épisode (rebond) OK
    both("\n4) Verdict global :")
    if coherence_ok:
        verdict = "Regime plausible, pret pour reentrainement du TFT"
        both(f">>> {verdict}")
        both(
            "    Les mois bear coïncident majoritairement avec des baisses MASI, "
            "les episodes sont groupes, et bull n'est pas systematiqueement incoherent."
        )
    else:
        verdict = "Incoherences detectees, calibration a revoir"
        both(f">>> {verdict}")
        problems = []
        if st_b["pct_neg_1m"] < 60:
            problems.append(
                f"seulement {st_b['pct_neg_1m']:.0f}% des bear ont un ret_1m negatif"
            )
        if st_b["ret_1m_mean"] >= 0:
            problems.append("ret_1m moyen bear non negatif")
        if not persist_ok:
            problems.append("episodes bear trop fragmentes")
        if st_u["n"] and st_u["pct_pos_1m"] < 50 and st_u["ret_1m_mean"] <= 0:
            problems.append("bull incoherent avec MASI")
        for p in problems:
            both(f"    - {p}")
        if len(bear_bad):
            both(
                "    - Bear a investiguer : "
                + ", ".join(bear_bad["month_label"].tolist())
            )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    both(f"\nRapport texte : {OUT_REPORT}")
    both("Diagnostic termine (lecture seule).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
