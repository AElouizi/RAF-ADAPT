"""
Diagnostic lecture seule — calibration règle a seuils (3 regimes) vs HMM.

Etapes 1-4 uniquement : seuils train 2015-2020, distribution, plausibilite MASI.
Ne modifie ni features_indices ni le TFT.

Usage :
  py -3 diagnostics/threshold_regime_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import FEATURES_DIR, SPLIT_TRAIN_END  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "threshold_regime_months.csv"
OUT_REPORT = OUT_DIR / "threshold_regime_report.txt"

# Noms projet : masi_mom_3m (pas masi_momentum_3m) ; is_neutral = is_sideways en prod.
MOM_COL = "masi_mom_3m"
BR_COL = "breadth_ma50"
SUBPERIODS = [
    (2015, 2017, "2015-2017"),
    (2018, 2020, "2018-2020"),
    (2021, 2022, "2021-2022"),
    (2023, 2025, "2023-2025"),
]


def _load_monthly_eom() -> pd.DataFrame:
    path = FEATURES_DIR / "features_indices.parquet"
    df = pd.read_parquet(path)
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    df[MOM_COL] = pd.to_numeric(df[MOM_COL], errors="coerce")
    df[BR_COL] = pd.to_numeric(df[BR_COL], errors="coerce")
    df = df.dropna(subset=["date_cours", MOM_COL, BR_COL]).sort_values("date_cours")
    df["year"] = df["date_cours"].dt.year
    df["month"] = df["date_cours"].dt.month
    df = df[(df["year"] >= 2015) & (df["year"] <= 2025)]

    eom = (
        df.groupby(["year", "month"], as_index=False)
        .tail(1)[
            [
                "year",
                "month",
                "date_cours",
                MOM_COL,
                BR_COL,
                "is_bull",
                "is_sideways",
                "is_bear",
            ]
        ]
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    eom["month_label"] = eom.apply(lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1)
    eom["hmm_regime"] = np.select(
        [
            eom["is_bull"].fillna(0).astype(int) == 1,
            eom["is_bear"].fillna(0).astype(int) == 1,
        ],
        ["bull", "bear"],
        default="neutral",
    )
    return eom


def _load_masi_eom() -> pd.DataFrame:
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
    return eom


def _calibrate_thresholds(
    train: pd.DataFrame, q_low: float, q_high: float
) -> dict[str, float]:
    return {
        "p_low_mom": float(train[MOM_COL].quantile(q_low)),
        "p_high_mom": float(train[MOM_COL].quantile(q_high)),
        "p_low_br": float(train[BR_COL].quantile(q_low)),
        "p_high_br": float(train[BR_COL].quantile(q_high)),
        "q_low": q_low,
        "q_high": q_high,
    }


def _apply_rule(
    df: pd.DataFrame,
    thr: dict[str, float],
    *,
    mode: str = "both",
) -> pd.DataFrame:
    """
    mode:
      both     — bull/bear exigent momentum ET breadth
      mom_only — uniquement momentum
      br_only  — uniquement breadth
      either   — bull si (mom>high OR br>high) AND NOT bear-like; bear symetrique
                 (en pratique: bull = mom>high & br>high reste; either = OR pour bull/bear
                  avec priorite bear si conflit)
    """
    out = df.copy()
    mom = out[MOM_COL]
    br = out[BR_COL]
    hi_m, lo_m = thr["p_high_mom"], thr["p_low_mom"]
    hi_b, lo_b = thr["p_high_br"], thr["p_low_br"]

    if mode == "both":
        is_bull = (mom > hi_m) & (br > hi_b)
        is_bear = (mom < lo_m) & (br < lo_b)
    elif mode == "mom_only":
        is_bull = mom > hi_m
        is_bear = mom < lo_m
    elif mode == "br_only":
        is_bull = br > hi_b
        is_bear = br < lo_b
    elif mode == "either":
        # bull si au moins une variable au-dessus du haut, et aucune en zone bear
        is_bull = ((mom > hi_m) | (br > hi_b)) & ~((mom < lo_m) | (br < lo_b))
        is_bear = ((mom < lo_m) | (br < lo_b)) & ~((mom > hi_m) | (br > hi_b))
        # conflit (une haute, une basse) → neutral
    else:
        raise ValueError(mode)

    # Conflit residuel (impossible en both/mom/br si mutuellement exclusifs, sauf either)
    conflict = is_bull & is_bear
    is_bull = is_bull & ~conflict
    is_bear = is_bear & ~conflict
    is_neutral = ~is_bull & ~is_bear

    out["is_bull"] = is_bull.astype(int)
    out["is_neutral"] = is_neutral.astype(int)
    out["is_bear"] = is_bear.astype(int)
    out["regime"] = np.select(
        [out["is_bull"] == 1, out["is_bear"] == 1],
        ["bull", "bear"],
        default="neutral",
    )
    assert (out[["is_bull", "is_neutral", "is_bear"]].sum(axis=1) == 1).all()
    return out


def _distribution(df: pd.DataFrame, label: str) -> list[str]:
    lines: list[str] = [f"\n=== Distribution — {label} (n={len(df)}) ==="]
    counts = df["regime"].value_counts()
    for r in ("bull", "neutral", "bear"):
        n = int(counts.get(r, 0))
        pct = 100.0 * n / len(df) if len(df) else 0.0
        lines.append(f"  {r:8s}: {n:3d} mois ({pct:5.1f}%)")

    lines.append("  Par sous-periode :")
    for y0, y1, name in SUBPERIODS:
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        if sub.empty:
            continue
        parts = []
        for r in ("bull", "neutral", "bear"):
            n = int((sub["regime"] == r).sum())
            pct = 100.0 * n / len(sub)
            parts.append(f"{r}={n} ({pct:.0f}%)")
        lines.append(f"    {name} (n={len(sub)}): " + " | ".join(parts))
    return lines


def _transitions_and_episodes(df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["\n=== Transitions & duree moyenne des episodes ==="]
    regimes = df["regime"].tolist()
    n_trans = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
    lines.append(f"  Nombre de transitions : {n_trans} / {len(df) - 1} mois adjacents")

    for r in ("bull", "neutral", "bear"):
        lengths: list[int] = []
        i = 0
        while i < len(regimes):
            if regimes[i] != r:
                i += 1
                continue
            j = i
            while j < len(regimes) and regimes[j] == r:
                j += 1
            lengths.append(j - i)
            i = j
        if lengths:
            lines.append(
                f"  {r:8s}: {len(lengths)} episodes | "
                f"duree moy={np.mean(lengths):.2f} | "
                f"med={np.median(lengths):.1f} | max={max(lengths)}"
            )
        else:
            lines.append(f"  {r:8s}: aucun episode")
    return lines


def _longest_episodes(df: pd.DataFrame, regime: str, k: int = 5) -> list[str]:
    lines: list[str] = [f"\n=== Episodes {regime} les plus longs (top {k}) ==="]
    regimes = df["regime"].tolist()
    labels = df["month_label"].tolist()
    rets = df["masi_ret_1m"].tolist() if "masi_ret_1m" in df.columns else [np.nan] * len(df)
    episodes: list[tuple[int, str, str, float]] = []
    i = 0
    while i < len(regimes):
        if regimes[i] != regime:
            i += 1
            continue
        j = i
        while j < len(regimes) and regimes[j] == regime:
            j += 1
        seg_rets = [x for x in rets[i:j] if pd.notna(x)]
        mean_ret = float(np.mean(seg_rets)) if seg_rets else float("nan")
        episodes.append((j - i, labels[i], labels[j - 1], mean_ret))
        i = j
    episodes.sort(key=lambda x: (-x[0], x[1]))
    for length, start, end, mean_ret in episodes[:k]:
        ret_s = "n/a" if pd.isna(mean_ret) else f"{100 * mean_ret:+.2f}%"
        lines.append(f"  {start} → {end} ({length} mois) | ret_1m moy={ret_s}")
    if not episodes:
        lines.append("  (aucun)")
    return lines


def _plausibility(df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["\n=== Plausibilite economique vs MASI (ret_1m) ==="]
    for r in ("bull", "neutral", "bear"):
        sub = df[df["regime"] == r].dropna(subset=["masi_ret_1m"])
        if sub.empty:
            lines.append(f"  {r}: n=0")
            continue
        mean = float(sub["masi_ret_1m"].mean())
        med = float(sub["masi_ret_1m"].median())
        pct_pos = 100.0 * float((sub["masi_ret_1m"] > 0).mean())
        pct_neg = 100.0 * float((sub["masi_ret_1m"] < 0).mean())
        lines.append(
            f"  {r:8s} n={len(sub):3d} | "
            f"mean={100 * mean:+.2f}% | med={100 * med:+.2f}% | "
            f"%pos={pct_pos:.1f}% | %neg={pct_neg:.1f}%"
        )
    bull = df[df["regime"] == "bull"].dropna(subset=["masi_ret_1m"])
    bear = df[df["regime"] == "bear"].dropna(subset=["masi_ret_1m"])
    if len(bull) and len(bear):
        ok_order = float(bull["masi_ret_1m"].mean()) > float(bear["masi_ret_1m"].mean())
        lines.append(
            f"\n  Ordre mean(bull) > mean(bear) : {'OUI' if ok_order else 'NON'}"
        )
    return lines


def _hmm_baseline(df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["\n=== Baseline HMM actuel (meme panel mensuel EOM) ==="]
    counts = df["hmm_regime"].value_counts()
    for r in ("bull", "neutral", "bear"):
        n = int(counts.get(r, 0))
        pct = 100.0 * n / len(df) if len(df) else 0.0
        lines.append(f"  {r:8s}: {n:3d} mois ({pct:5.1f}%)")
    return lines


def _min_class_pct(df: pd.DataFrame) -> float:
    return 100.0 * df["regime"].value_counts().min() / len(df)


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CALIBRATION REGIME A SEUILS (etapes 1-4)")
    lines.append(f"Seuils fit UNIQUEMENT sur train <= {SPLIT_TRAIN_END} (annees >= 2015)")
    lines.append("Colonnes : masi_mom_3m, breadth_ma50 (EOM mensuel)")
    lines.append("Encodage : is_bull / is_neutral / is_bear (somme=1)")
    lines.append("=" * 72)

    monthly = _load_monthly_eom()
    masi = _load_masi_eom()
    monthly = monthly.merge(masi[["year", "month", "masi_ret_1m"]], on=["year", "month"], how="left")

    train_end = pd.Timestamp(SPLIT_TRAIN_END)
    train = monthly[
        (monthly["year"] >= 2015)
        & (pd.to_datetime(monthly["date_cours"]) <= train_end)
    ].copy()
    lines.append(f"\nPanel mensuel 2015-2025 : {len(monthly)} mois")
    lines.append(f"Train 2015-2020 : {len(train)} mois")
    lines.extend(_hmm_baseline(monthly))

    # --- Variantes a comparer ---
    candidates: list[tuple[str, float, float, str]] = [
        ("both_33_67", 0.33, 0.67, "both"),
        ("both_25_75", 0.25, 0.75, "both"),
        ("both_40_60", 0.40, 0.60, "both"),
        ("mom_33_67", 0.33, 0.67, "mom_only"),
        ("br_33_67", 0.33, 0.67, "br_only"),
        ("either_33_67", 0.33, 0.67, "either"),
        ("either_25_75", 0.25, 0.75, "either"),
        ("mom_25_75", 0.25, 0.75, "mom_only"),
    ]

    summaries: list[dict] = []
    detailed: dict[str, pd.DataFrame] = {}
    thr_by_name: dict[str, dict] = {}

    for name, q_low, q_high, mode in candidates:
        thr = _calibrate_thresholds(train, q_low, q_high)
        thr_by_name[name] = thr
        labeled = _apply_rule(monthly, thr, mode=mode)
        detailed[name] = labeled
        summaries.append(
            {
                "name": name,
                "mode": mode,
                "q_low": q_low,
                "q_high": q_high,
                "min_pct": _min_class_pct(labeled),
                **{
                    r: 100.0 * (labeled["regime"] == r).mean()
                    for r in ("bull", "neutral", "bear")
                },
            }
        )

    lines.append("\n=== Comparaison des variantes (objectifs : min classe >= 10-15%) ===")
    lines.append(
        f"  {'variante':<16} {'mode':<10} {'q':>7} "
        f"{'bull%':>7} {'neut%':>7} {'bear%':>7} {'min%':>7}"
    )
    for s in summaries:
        lines.append(
            f"  {s['name']:<16} {s['mode']:<10} "
            f"{s['q_low']:.2f}/{s['q_high']:.2f} "
            f"{s['bull']:6.1f}% {s['neutral']:6.1f}% {s['bear']:6.1f}% "
            f"{s['min_pct']:6.1f}%"
        )

    # Selection : priorite both_33_67 si OK, sinon meilleure min_pct parmi both*,
    # sinon meilleure globale. Preferer "both" pour interpretation economique.
    primary = "both_33_67"
    primary_df = detailed[primary]
    primary_ok = _min_class_pct(primary_df) >= 10.0

    if primary_ok:
        chosen = primary
        reason = "Regle canonique both AND (p33/p67) : toutes classes >= 10%."
    else:
        # Prefer both* with min>=10, else either/mom with highest min
        both_ok = [
            s for s in summaries if s["mode"] == "both" and s["min_pct"] >= 10.0
        ]
        if both_ok:
            chosen = max(both_ok, key=lambda s: s["min_pct"])["name"]
            reason = (
                f"{primary} trop desequilibree (min={_min_class_pct(primary_df):.1f}%). "
                f"Retention variante both avec meilleure min classe : {chosen}."
            )
        else:
            # Prefer variants with economic AND logic softened, or single-var
            pool = [s for s in summaries if s["min_pct"] >= 10.0]
            if not pool:
                pool = summaries
            # Prefer mom_only then either then br_only
            mode_rank = {"mom_only": 0, "either": 1, "br_only": 2, "both": 3}
            chosen = sorted(
                pool,
                key=lambda s: (-s["min_pct"], mode_rank.get(s["mode"], 9), s["name"]),
            )[0]["name"]
            reason = (
                f"Aucune variante both n'atteint 10%. "
                f"Retention : {chosen} (meilleure distribution / interpretabilite)."
            )

    lines.append(f"\n>>> VARIANTE RETENUE POUR DETAIL : {chosen}")
    lines.append(f"    Motif : {reason}")

    thr = thr_by_name[chosen]
    labeled = detailed[chosen]

    lines.append("\n=== ETAPE 1 — Seuils calibres sur train 2015-2020 ===")
    lines.append(f"  Variante : {chosen} | mode={thr_by_name[chosen] and next(s['mode'] for s in summaries if s['name']==chosen)}")
    lines.append(f"  Quantiles : q_low={thr['q_low']}, q_high={thr['q_high']}")
    lines.append(f"  p_low_momentum  (q={thr['q_low']})  = {thr['p_low_mom']:.6f}")
    lines.append(f"  p_high_momentum (q={thr['q_high']}) = {thr['p_high_mom']:.6f}")
    lines.append(f"  p_low_breadth   (q={thr['q_low']})  = {thr['p_low_br']:.6f}")
    lines.append(f"  p_high_breadth  (q={thr['q_high']}) = {thr['p_high_br']:.6f}")

    # Always also print the canonical 33/67 both thresholds explicitly
    thr3367 = thr_by_name["both_33_67"]
    lines.append("\n  [Reference canonique both_33_67 — toujours documentee]")
    lines.append(f"  p33_momentum = {thr3367['p_low_mom']:.6f}")
    lines.append(f"  p67_momentum = {thr3367['p_high_mom']:.6f}")
    lines.append(f"  p33_breadth  = {thr3367['p_low_br']:.6f}")
    lines.append(f"  p67_breadth  = {thr3367['p_high_br']:.6f}")

    lines.append("\n=== ETAPE 2 — Regle appliquee (one-hot, somme=1) ===")
    mode_chosen = next(s["mode"] for s in summaries if s["name"] == chosen)
    if mode_chosen == "both":
        lines.append("  is_bull    = (mom > p_high_mom) & (br > p_high_br)")
        lines.append("  is_bear    = (mom < p_low_mom)  & (br < p_low_br)")
        lines.append("  is_neutral = ~is_bull & ~is_bear")
    elif mode_chosen == "mom_only":
        lines.append("  is_bull    = (mom > p_high_mom)")
        lines.append("  is_bear    = (mom < p_low_mom)")
        lines.append("  is_neutral = ~is_bull & ~is_bear")
    elif mode_chosen == "either":
        lines.append("  is_bull    = (mom>high | br>high) & ~(mom<low | br<low)")
        lines.append("  is_bear    = (mom<low  | br<low)  & ~(mom>high | br>high)")
        lines.append("  is_neutral = reste (y compris signaux contradictoires)")
    else:
        lines.append(f"  mode={mode_chosen}")

    srow = labeled[["is_bull", "is_neutral", "is_bear"]].sum(axis=1)
    lines.append(f"  Verification somme==1 : {(srow == 1).all()} (n={len(labeled)})")

    lines.extend(_distribution(labeled, chosen))
    lines.extend(_transitions_and_episodes(labeled))
    lines.extend(_plausibility(labeled))
    lines.extend(_longest_episodes(labeled, "bear", k=6))
    lines.extend(_longest_episodes(labeled, "bull", k=5))

    # COVID check
    covid = labeled[labeled["month_label"].isin(["2020-03", "2020-04", "2020-05"])]
    lines.append("\n=== Check COVID (mars-mai 2020) ===")
    for _, r in covid.iterrows():
        ret = r.get("masi_ret_1m")
        ret_s = "n/a" if pd.isna(ret) else f"{100 * ret:+.2f}%"
        lines.append(f"  {r['month_label']}: regime={r['regime']} | MASI={ret_s}")

    # Also detail both_33_67 if not chosen
    if chosen != "both_33_67":
        lines.append("\n" + "=" * 72)
        lines.append("DETAIL COMPLEMENTAIRE — variante canonique both_33_67")
        lines.append("=" * 72)
        alt = detailed["both_33_67"]
        lines.extend(_distribution(alt, "both_33_67"))
        lines.extend(_transitions_and_episodes(alt))
        lines.extend(_plausibility(alt))
        lines.extend(_longest_episodes(alt, "bear", k=6))
        covid2 = alt[alt["month_label"].isin(["2020-03", "2020-04", "2020-05"])]
        lines.append("\n=== Check COVID both_33_67 ===")
        for _, r in covid2.iterrows():
            ret = r.get("masi_ret_1m")
            ret_s = "n/a" if pd.isna(ret) else f"{100 * ret:+.2f}%"
            lines.append(f"  {r['month_label']}: regime={r['regime']} | MASI={ret_s}")

    # Save chosen monthly panel
    export_cols = [
        "month_label",
        "year",
        "month",
        "date_cours",
        MOM_COL,
        BR_COL,
        "is_bull",
        "is_neutral",
        "is_bear",
        "regime",
        "hmm_regime",
        "masi_ret_1m",
    ]
    labeled[export_cols].to_csv(OUT_CSV, index=False)

    text = "\n".join(lines) + "\n"
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"CSV  → {OUT_CSV}")
    print(f"Report → {OUT_REPORT}")


if __name__ == "__main__":
    main()
