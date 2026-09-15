"""
Diagnostic lecture seule — variabilité du régime de marché (Composante A, HMM).

Charge features_indices (Supabase ou parquet local), recalcule le HMM actuel,
affiche stats / distribution / transitions, et écrit diagnostics/regime_timeline.png.

Usage :
  py -3 diagnostics/regime_variability_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    FEATURES_DIR,
    PROJECT_ROOT,
    REGIME_END_YEAR,
    REGIME_START_YEAR,
    SPLIT_TRAIN_END,
    TABLE_FEATURES_INDICES,
    load_env_file,
)
from bvc_recommender.models.regime_detector import (  # noqa: E402
    HMMRegimeDetector,
    REGIME_COLUMNS,
    REGIME_INPUTS,
    detect_regime_history,
    dominant_regime,
)

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "regime_timeline.png"

INPUT_COLS = list(REGIME_INPUTS)  # masi_mom_3m, return_dispersion, breadth_ma50
SUBPERIODS = [
    (2015, 2017, "2015-2017"),
    (2018, 2020, "2018-2020"),
    (2021, 2022, "2021-2022"),
    (2023, 2025, "2023-2025"),
]
REGIME_COLORS = {"bull": "#2ca02c", "sideways": "#ff7f0e", "bear": "#d62728"}
REGIME_LABELS_FR = {"bull": "Haussier", "sideways": "Latéral", "bear": "Baissier"}


def _load_features_indices() -> tuple[pd.DataFrame, str]:
    """Charge le panel quotidien : parquet local → Postgres → REST Supabase."""
    load_env_file()

    for ext in (".parquet", ".csv"):
        path = FEATURES_DIR / f"features_indices{ext}"
        if path.is_file():
            df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
            return df, f"local:{path}"

    from bvc_recommender.data.loader import (
        fetch_table,
        fetch_table_postgres,
        get_supabase_client,
    )

    pg = fetch_table_postgres(TABLE_FEATURES_INDICES)
    if pg is not None and not pg.empty:
        return pg, f"postgres:{TABLE_FEATURES_INDICES}"

    client = get_supabase_client()
    cols = ",".join(["date_cours", *INPUT_COLS])
    # order_by date_cours (pas id — table indices)
    try:
        df = fetch_table(
            client,
            TABLE_FEATURES_INDICES,
            columns=cols,
            order_by="date_cours",
            page_size=1000,
        )
    except Exception:
        df = fetch_table(
            client,
            TABLE_FEATURES_INDICES,
            columns="*",
            order_by="date_cours",
            page_size=500,
        )
    if df is None or df.empty:
        raise FileNotFoundError(
            "Impossible de charger features_indices (parquet absent + Supabase vide)."
        )
    return df, f"supabase_rest:{TABLE_FEATURES_INDICES}"


def _describe(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    qs = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "p10": float(qs.loc[0.10]),
        "p25": float(qs.loc[0.25]),
        "p50": float(qs.loc[0.50]),
        "p75": float(qs.loc[0.75]),
        "p90": float(qs.loc[0.90]),
    }


def _print_desc(name: str, stats: dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  n={stats['n']}  min={stats['min']:.6f}  max={stats['max']:.6f}  "
        f"mean={stats['mean']:.6f}  std={stats['std']:.6f}"
    )
    print(
        f"  p10={stats['p10']:.6f}  p25={stats['p25']:.6f}  p50={stats['p50']:.6f}  "
        f"p75={stats['p75']:.6f}  p90={stats['p90']:.6f}"
    )


def _regime_counts(monthly: pd.DataFrame) -> pd.Series:
    return monthly["regime"].value_counts().reindex(["bull", "sideways", "bear"], fill_value=0)


def _print_distribution(title: str, counts: pd.Series) -> None:
    total = int(counts.sum())
    print(f"\n--- {title} (n={total} mois) ---")
    for lab in ("bull", "sideways", "bear"):
        n = int(counts.get(lab, 0))
        pct = 100.0 * n / total if total else 0.0
        print(f"  {REGIME_LABELS_FR[lab]:10s} ({lab:8s}) : {n:4d}  ({pct:5.1f}%)")


def _print_fuzzy_vs_percentiles(stats_by_col: dict[str, dict[str, float]]) -> None:
    """Note comparative (ablation) — bornes trimf vs percentiles historiques."""
    from bvc_recommender.benchmarking.fuzzy_regime import FuzzyRegimeDetector

    f = FuzzyRegimeDetector()
    mapping = {
        "masi_mom_3m": [
            ("mom_neg", f.mom_neg),
            ("mom_neu", f.mom_neu),
            ("mom_pos", f.mom_pos),
        ],
        "breadth_ma50": [
            ("br_low", f.br_low),
            ("br_med", f.br_med),
            ("br_high", f.br_high),
        ],
        "return_dispersion": [
            ("disp_low", f.disp_low),
            ("disp_med", f.disp_med),
            ("disp_high", f.disp_high),
        ],
    }
    print("\n=== SENSIBILITÉ (référence ablation floue — NON utilisé en production) ===")
    print("Bornes trimf vs percentiles historiques (lecture seule) :")
    for col, triangles in mapping.items():
        st = stats_by_col[col]
        print(f"\n  [{col}] historique p10={st['p10']:.4f} p50={st['p50']:.4f} p90={st['p90']:.4f}")
        for name, tri in triangles:
            a, b, c = tri
            # percentile empirique approximatif du sommet c (borne haute)
            s = None  # filled below only for messaging
            print(f"    {name}: trimf{tri}")


def _empirical_percentile(value: float, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float((s <= value).mean() * 100.0)


def _scaled_predict(det: HMMRegimeDetector, X_raw: np.ndarray) -> np.ndarray:
    """Viterbi sur features standardisées (scaler fit au train)."""
    if det._scaler is None or det._model is None:
        raise RuntimeError("HMM non entraîné")
    return det._model.predict(det._scaler.transform(X_raw))


def _means_raw(det: HMMRegimeDetector) -> dict[int, np.ndarray]:
    """Moyennes d'état ramenées à l'échelle brute des features."""
    assert det._model is not None
    means_scaled = det._model.means_
    if det._scaler is not None:
        means = det._scaler.inverse_transform(means_scaled)
    else:
        means = means_scaled
    return {s: means[s] for s in range(det.n_components)}


def _print_hmm_params(
    det: HMMRegimeDetector,
    train_df: pd.DataFrame,
    full_daily_states: np.ndarray,
) -> None:
    model = det._model
    assert model is not None
    print("\n=== PARAMÈTRES HMM APPRIS (production) ===")
    print(
        f"  train_end = {det.train_end}  |  n_components={det.n_components}  "
        f"| min_covar={getattr(det, 'min_covar', None)}  "
        f"| scaler={'StandardScaler' if det._scaler is not None else 'none'}"
    )
    print(f"  state → label : {det._state_to_label}")
    means_raw = _means_raw(det)
    print("\n  Moyennes BRUTES par état [masi_mom_3m, return_dispersion, breadth_ma50] :")
    for s in range(det.n_components):
        lab = det._state_to_label.get(s, "?")
        means = means_raw[s]
        vars_ = np.diag(model.covars_[s]) if model.covars_[s].ndim == 2 else model.covars_[s]
        print(
            f"    état {s} ({lab:8s})  mean={np.round(means, 6).tolist()}  "
            f"var_scaled={np.round(vars_, 8).tolist()}"
        )

    # Obs train assignées
    X_train = train_df[INPUT_COLS].to_numpy(dtype=float)
    states_train = _scaled_predict(det, X_train)
    print("\n  Observations TRAIN assignées (Viterbi ≤ train_end) :")
    for s in range(det.n_components):
        lab = det._state_to_label.get(s, "?")
        n = int(np.sum(states_train == s))
        pct = 100.0 * n / len(states_train) if len(states_train) else 0.0
        print(f"    état {s} ({lab:8s}) : {n:5d} j  ({pct:5.1f}%)")

    print("\n  Observations FULL période assignées (Viterbi 2015–2025 journalier) :")
    for s in range(det.n_components):
        lab = det._state_to_label.get(s, "?")
        n = int(np.sum(full_daily_states == s))
        pct = 100.0 * n / len(full_daily_states) if len(full_daily_states) else 0.0
        print(f"    état {s} ({lab:8s}) : {n:5d} j  ({pct:5.1f}%)")


def _plot_timeline(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    det: HMMRegimeDetector,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    assert det._model is not None

    # Moyennes par label en échelle brute
    means_raw = _means_raw(det)
    means_by_label: dict[str, np.ndarray] = {}
    for s, lab in det._state_to_label.items():
        means_by_label[lab] = means_raw[s]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "Diagnostic régime HMM — Composante A (2015–2025)",
        fontsize=13,
        fontweight="bold",
    )

    # 1) Régime mensuel
    ax0 = axes[0]
    code = {"bear": 0, "sideways": 1, "bull": 2}
    y = monthly["regime"].map(code).astype(float)
    t = pd.to_datetime(monthly["as_of_date"])
    ax0.step(t, y, where="mid", color="#1f77b4", linewidth=1.4)
    ax0.scatter(t, y, c=monthly["regime"].map(REGIME_COLORS), s=18, zorder=3)
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(["Bear", "Sideways", "Bull"])
    ax0.set_ylabel("Régime")
    ax0.set_title("Régime détecté (dernier jour de bourse du mois)")
    ax0.grid(True, alpha=0.3)

    # Color bands
    for i in range(len(monthly) - 1):
        ax0.axvspan(
            t.iloc[i],
            t.iloc[i + 1],
            facecolor=REGIME_COLORS[monthly["regime"].iloc[i]],
            alpha=0.12,
            linewidth=0,
        )

    # 2–4) Features journalières + moyennes d'état
    feature_axes = [
        (axes[1], "masi_mom_3m", "Momentum MASI 3M"),
        (axes[2], "return_dispersion", "Dispersion des rendements"),
        (axes[3], "breadth_ma50", "Breadth MA50"),
    ]
    d = daily.copy()
    d["date_cours"] = pd.to_datetime(d["date_cours"])
    d = d.sort_values("date_cours")

    for ax, col, title in feature_axes:
        ax.plot(d["date_cours"], d[col], color="#555555", linewidth=0.7, alpha=0.85)
        for lab, means in means_by_label.items():
            idx = INPUT_COLS.index(col)
            ax.axhline(
                means[idx],
                color=REGIME_COLORS[lab],
                linestyle="--",
                linewidth=1.2,
                label=f"μ {lab}={means[idx]:.4f}",
            )
        ax.set_ylabel(col)
        ax.set_title(title + " + moyennes d'état HMM (train)")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure écrite : {out_path}")


def _summary_answers(
    monthly: pd.DataFrame,
    counts_all: pd.Series,
    counts_by_period: dict[str, pd.Series],
    det: HMMRegimeDetector,
    stats_by_col: dict[str, dict[str, float]],
    daily: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("RÉSUMÉ — RÉPONSES AUX QUESTIONS DU DIAGNOSTIC")
    print("=" * 72)

    bear_months = monthly.loc[monthly["regime"] == "bear", "month_label"].tolist()
    print("\n1) Le régime BEAR est-il jamais détecté sur 2015–2025 ?")
    if bear_months:
        print(f"   OUI — {len(bear_months)} mois baissiers :")
        # group consecutive for readability
        print("   " + ", ".join(bear_months))
    else:
        print("   NON — aucun mois classé bear sur toute la période.")

    total = int(counts_all.sum())
    side_all = int(counts_all.get("sideways", 0))
    side_pct_all = 100.0 * side_all / total if total else 0.0
    c2325 = counts_by_period.get("2023-2025", pd.Series(dtype=int))
    tot2325 = int(c2325.sum()) if len(c2325) else 0
    side2325 = int(c2325.get("sideways", 0)) if tot2325 else 0
    side_pct_2325 = 100.0 * side2325 / tot2325 if tot2325 else 0.0

    print("\n2) Le régime dominant (sideways) représente-t-il >80% des mois ?")
    print(f"   Ensemble 2015–2025 : sideways = {side_all}/{total} = {side_pct_all:.1f}%")
    print(f"   Sous-période 2023–2025 : sideways = {side2325}/{tot2325} = {side_pct_2325:.1f}%")
    if side_pct_all > 80:
        print("   → OUI, dominance sideways GLOBALE (>80% sur toute la période).")
    elif side_pct_2325 > 80:
        print("   → Dominance sideways surtout sur 2023–2025 (>80%), pas forcément globale.")
    else:
        print("   → NON, sideways ne dépasse pas 80% (ni globalement, ni sur 2023–2025).")

    # Dominant overall
    dominant = counts_all.idxmax()
    dom_pct = 100.0 * int(counts_all.max()) / total if total else 0.0
    print(f"   Régime le plus fréquent globalement : {dominant} ({dom_pct:.1f}%).")

    print("\n3) Les paramètres actuels semblent-ils mal calibrés ?")
    print("   (HMM : pas de seuils fixes — on compare moyennes d'état vs distribution réelle)")
    assert det._model is not None
    issues: list[str] = []
    means_raw = _means_raw(det)
    for s, lab in det._state_to_label.items():
        means = means_raw[s]
        for j, col in enumerate(INPUT_COLS):
            mu = float(means[j])
            st = stats_by_col[col]
            # hors [min, max] historique
            if mu < st["min"] or mu > st["max"]:
                issues.append(
                    f"μ_{lab}.{col}={mu:.4f} hors [min,max]=[{st['min']:.4f},{st['max']:.4f}]"
                )
            pct = _empirical_percentile(mu, daily[col])
            # moyenne d'état extrême (>p95 ou <p5)
            if pct >= 95 or pct <= 5:
                issues.append(
                    f"μ_{lab}.{col}={mu:.4f} ≈ percentile empirique {pct:.1f}% "
                    f"(extrême vs historique)"
                )

    # Check state occupancy imbalance
    full_states = _scaled_predict(det, daily[INPUT_COLS].to_numpy(dtype=float))
    for s, lab in det._state_to_label.items():
        share = float(np.mean(full_states == s))
        if share < 0.05:
            issues.append(f"État {lab} très rare en journalier ({100*share:.1f}% des jours)")
        if share > 0.85:
            issues.append(f"État {lab} quasi-permanent en journalier ({100*share:.1f}% des jours)")

    # Monthly imbalance
    if side_pct_all > 80:
        issues.append(
            f"Sideways mensuel = {side_pct_all:.1f}% → signal de régime quasi-constant, "
            "peu informatif pour un TFT conditionné"
        )

    if issues:
        print("   Signaux d'alerte :")
        for msg in issues:
            print(f"   - {msg}")
        print("   → Calibration / séparabilité des états à revoir (pas de seuil flou fixe).")
    else:
        print("   Pas d'anomalie grossière (moyennes d'état dans la plage historique,")
        print("   occupancy raisonnable). La faible variabilité peut venir de la BVC elle-même")
        print("   ou d'une mauvaise séparabilité des 3 variables d'entrée.")

    # Fuzzy note
    print("\n   Note : les bornes floues (ablation) sont documentées plus haut ;")
    print("   la production utilise exclusivement le HMM (pas ces seuils).")


def main() -> int:
    print("=" * 72)
    print("DIAGNOSTIC VARIABILITÉ RÉGIME — Composante A (HMM production)")
    print("=" * 72)

    ctx, source = _load_features_indices()
    ctx["date_cours"] = pd.to_datetime(ctx["date_cours"], errors="coerce")
    for c in INPUT_COLS:
        if c not in ctx.columns:
            raise KeyError(f"Colonne manquante dans features_indices : {c}")
        ctx[c] = pd.to_numeric(ctx[c], errors="coerce")

    ctx = ctx.dropna(subset=["date_cours", *INPUT_COLS]).sort_values("date_cours")
    ctx = ctx.drop_duplicates("date_cours", keep="last").reset_index(drop=True)
    # Ne pas tronquer avant Viterbi : le décodage global doit matcher la prod (step4).
    # Le filtre 2015–2025 s'applique à l'agrégation mensuelle / stats affichées.

    print(f"\nSource données : {source}")
    print(f"Periode panel  : {ctx['date_cours'].min().date()} -> {ctx['date_cours'].max().date()}")
    print(f"Jours panel    : {len(ctx)}")
    print(f"Colonnes       : {INPUT_COLS}")
    print(f"Train HMM      : {REGIME_START_YEAR} → {SPLIT_TRAIN_END}")

    ctx_win = ctx[
        (ctx["date_cours"].dt.year >= REGIME_START_YEAR)
        & (ctx["date_cours"].dt.year <= REGIME_END_YEAR)
    ].copy()

    # 1) Stats descriptives
    stats_by_col = {c: _describe(ctx_win[c]) for c in INPUT_COLS}
    print("\n" + "=" * 72)
    print("1. STATISTIQUES DESCRIPTIVES (journalier, 2015–2025)")
    print("=" * 72)
    for c in INPUT_COLS:
        _print_desc(c, stats_by_col[c])

    # 2) Fit HMM + historique mensuel (Viterbi sur panel complet)
    det = HMMRegimeDetector(train_end=SPLIT_TRAIN_END)
    det.fit(ctx)
    monthly = detect_regime_history(
        ctx,
        start_year=REGIME_START_YEAR,
        end_year=REGIME_END_YEAR,
        detector=det,
    )
    if monthly.empty:
        print("ERREUR : aucun mois de régime calculé.")
        return 1
    monthly["regime"] = monthly.apply(dominant_regime, axis=1)

    # Daily panel for plot + state counts
    from bvc_recommender.models.regime_detector import detect_regime_daily

    daily = detect_regime_daily(
        ctx,
        start_year=REGIME_START_YEAR,
        end_year=REGIME_END_YEAR,
        detector=det,
    )
    X_full = daily[INPUT_COLS].to_numpy(dtype=float)
    full_states = _scaled_predict(det, X_full)

    train_mask = (
        (ctx["date_cours"].dt.year >= REGIME_START_YEAR)
        & (ctx["date_cours"] <= pd.Timestamp(SPLIT_TRAIN_END))
    )
    train_df = ctx.loc[train_mask].copy()

    print("\n" + "=" * 72)
    print("2. DISTRIBUTION DU RÉGIME MENSUEL")
    print("=" * 72)
    counts_all = _regime_counts(monthly)
    _print_distribution("Ensemble 2015–2025", counts_all)

    # Comparaison explicite vs ancien diagnostic (pre-correction splits)
    OLD = {"bull": 4, "sideways": 127, "bear": 0, "n": 131, "transitions": 8}
    print("\n--- Comparaison AVANT (pre-correction) vs APRÈS ---")
    print(f"{'regime':<12} {'AVANT':>8} {'APRÈS':>8} {'delta':>8}")
    for lab in ("bull", "sideways", "bear"):
        b = OLD[lab]
        a = int(counts_all.get(lab, 0))
        print(f"{lab:<12} {b:8d} {a:8d} {a-b:+8d}")
    print(
        f"{'total mois':<12} {OLD['n']:8d} {int(counts_all.sum()):8d} "
        f"{int(counts_all.sum())-OLD['n']:+8d}"
    )

    counts_by_period: dict[str, pd.Series] = {}
    for y0, y1, label in SUBPERIODS:
        sub = monthly[(monthly["year"] >= y0) & (monthly["year"] <= y1)]
        counts_by_period[label] = _regime_counts(sub)
        _print_distribution(f"Sous-période {label}", counts_by_period[label])

    # Transitions
    print("\n" + "=" * 72)
    print("3. TRANSITIONS DE RÉGIME (mois → mois suivant)")
    print("=" * 72)
    regs = monthly["regime"].tolist()
    labels = monthly["month_label"].tolist()
    transitions = []
    for i in range(1, len(regs)):
        if regs[i] != regs[i - 1]:
            transitions.append((labels[i - 1], regs[i - 1], labels[i], regs[i]))
    n_months = len(regs)
    n_trans = len(transitions)
    freq = n_trans / (n_months - 1) if n_months > 1 else 0.0
    print(f"  Mois observés     : {n_months}")
    print(f"  Transitions       : {n_trans}  (avant correction splits : {OLD['transitions']})")
    print(f"  Fréquence         : {freq:.3f} ({100*freq:.1f}% des pas mensuels)")
    if transitions:
        print("  Détail :")
        for a, ra, b, rb in transitions:
            print(f"    {a} ({ra}) → {b} ({rb})")
    else:
        print("  Aucune transition — régime constant sur toute la période.")

    # HMM params + fuzzy comparison
    print("\n" + "=" * 72)
    print("4. SENSIBILITÉ / PARAMÈTRES")
    print("=" * 72)
    _print_hmm_params(det, train_df, full_states)

    print("\n--- Comparaison moyennes HMM (échelle brute) vs percentiles historiques ---")
    means_raw = _means_raw(det)
    for s, lab in det._state_to_label.items():
        means = means_raw[s]
        for j, col in enumerate(INPUT_COLS):
            mu = float(means[j])
            pct = _empirical_percentile(mu, ctx[col])
            st = stats_by_col[col]
            print(
                f"  μ_{lab}.{col} = {mu:.6f}  → percentile empirique ≈ {pct:.1f}%  "
                f"(p50={st['p50']:.6f})"
            )

    try:
        _print_fuzzy_vs_percentiles(stats_by_col)
        from bvc_recommender.benchmarking.fuzzy_regime import FuzzyRegimeDetector

        f = FuzzyRegimeDetector()
        print("\n  Percentiles des bornes hautes floues (c du trimf) :")
        for col, tri, name in [
            ("masi_mom_3m", f.mom_pos, "mom_pos.c"),
            ("masi_mom_3m", f.mom_neg, "mom_neg.a"),
            ("breadth_ma50", f.br_high, "br_high.a"),
            ("breadth_ma50", f.br_low, "br_low.c"),
            ("return_dispersion", f.disp_high, "disp_high.c"),
            ("return_dispersion", f.disp_low, "disp_low.c"),
        ]:
            edge = tri[2] if "high" in name or name.endswith(".c") else tri[0]
            if name.endswith(".a") and "neg" in name:
                edge = tri[0]
            if name == "br_high.a":
                edge = tri[0]
            pct = _empirical_percentile(edge, ctx[col])
            print(f"    {name}={edge:.4f} sur {col} → percentile empirique ≈ {pct:.1f}%")
    except Exception as exc:
        print(f"\n(Note floue indisponible : {exc})")

    # Plot
    print("\n" + "=" * 72)
    print("5. VISUALISATION")
    print("=" * 72)
    _plot_timeline(monthly, daily, det, OUT_PNG)

    # Final Q&A
    _summary_answers(monthly, counts_all, counts_by_period, det, stats_by_col, daily)

    print("\nDiagnostic terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
