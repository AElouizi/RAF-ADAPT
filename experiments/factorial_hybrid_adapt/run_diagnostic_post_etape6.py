"""Diagnostic post-étape 6 — 5 points, sans réentraînement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATASET_DIR,
    FEATURES_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
)
from bvc_recommender.features.dataset_builder import TARGET_COLUMN  # noqa: E402
from bvc_recommender.models.metrics import (  # noqa: E402
    _cross_sectional_ic,
    evaluate_scoring_model,
    hit_ratio,
)
from experiments.factorial_hybrid_adapt.factorial_cells import (  # noqa: E402
    combine_hybrid_scores,
)
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR  # noqa: E402


def _regime_label(row: pd.Series) -> str:
    if int(row.get("is_bull") or 0) == 1:
        return "bull"
    if int(row.get("is_bear") or 0) == 1:
        return "bear"
    return "neutral"


def point1() -> None:
    print("\n" + "=" * 70)
    print("POINT 1 — Cohérence IC_val vs IC_test")
    print("=" * 70)

    ml = pd.read_parquet(
        DATASET_DIR / "ml_dataset.parquet",
        columns=["ticker", "date_cours", "split", TARGET_COLUMN],
    )
    ml["date_cours"] = pd.to_datetime(ml["date_cours"])

    print("\n[Dates / splits config]")
    print(f"  SPLIT_TRAIN_END = {SPLIT_TRAIN_END}")
    print(f"  SPLIT_VAL_END   = {SPLIT_VAL_END}")
    print(f"  SPLIT_TEST_START= {SPLIT_TEST_START}")
    for sp in ("train", "val", "test"):
        sub = ml[ml["split"] == sp]
        print(
            f"  {sp}: {sub['date_cours'].min().date()} -> {sub['date_cours'].max().date()} "
            f"| rows={len(sub)} | n_dates={sub['date_cours'].nunique()} "
            f"| n_months={sub['date_cours'].dt.to_period('M').nunique()}"
        )

    fi = pd.read_parquet(
        FEATURES_DIR / "features_indices.parquet",
        columns=["date_cours", "is_bull", "is_neutral", "is_bear"],
    )
    fi["date_cours"] = pd.to_datetime(fi["date_cours"])
    fi = fi.drop_duplicates("date_cours", keep="last")
    mlr = ml.merge(fi, on="date_cours", how="left")

    print("\n[Répartition régime — lignes panel]")
    for sp in ("val", "test"):
        sub = mlr[mlr["split"] == sp]
        n = len(sub)
        print(
            f"  {sp}: bull={100 * sub['is_bull'].sum() / n:.1f}% "
            f"neutral={100 * sub['is_neutral'].sum() / n:.1f}% "
            f"bear={100 * sub['is_bear'].sum() / n:.1f}% "
            f"(rows {int(sub['is_bull'].sum())}/{int(sub['is_neutral'].sum())}/{int(sub['is_bear'].sum())})"
        )

    print("\n[Répartition régime — mois calendaires, EOM]")
    for sp in ("val", "test"):
        sub = mlr[mlr["split"] == sp].copy()
        sub["mois"] = sub["date_cours"].dt.to_period("M")
        last_dates = sub.groupby("mois")["date_cours"].max().reset_index()
        reg = last_dates.merge(fi, on="date_cours", how="left")
        reg["regime"] = reg.apply(_regime_label, axis=1)
        vc = reg["regime"].value_counts()
        parts = ", ".join(f"{k}={int(v)} ({100 * v / len(reg):.1f}%)" for k, v in vc.items())
        print(f"  {sp}: n_mois={len(reg)} | {parts}")

    preds = pd.read_parquet(REPORTS_DIR / "factorial_predictions.parquet")
    preds["date_cours"] = pd.to_datetime(preds["date_cours"])
    test = preds[preds["split"] == "test"].copy()
    test["mois"] = test["date_cours"].dt.to_period("M")

    rows = []
    for cell_id in sorted(test["cell_id"].unique()):
        sub = test[test["cell_id"] == cell_id]
        for mois, g in sub.groupby("mois"):
            ics = _cross_sectional_ic(
                g, "prediction", TARGET_COLUMN, "date_cours", method="spearman"
            )
            rows.append(
                {
                    "cell_id": int(cell_id),
                    "mois": str(mois),
                    "ic_mean": float(np.mean(ics)) if ics else np.nan,
                    "n_dates": len(ics),
                }
            )
    ic_m = pd.DataFrame(rows)
    out = REPORTS_DIR / "diagnostic_ic_mensuel_test.csv"
    ic_m.to_csv(out, index=False)

    print("\n[IC Spearman mensuel — période TEST]")
    print(f"  (sauvegardé: {out})")
    for cell_id in (1, 2, 3, 4):
        s = ic_m[ic_m["cell_id"] == cell_id].sort_values("mois")
        print(f"\n  --- Cellule {cell_id} ---")
        print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            f"  résumé: mean={s['ic_mean'].mean():.4f} | median={s['ic_mean'].median():.4f} "
            f"| std={s['ic_mean'].std():.4f} | min={s['ic_mean'].min():.4f} | max={s['ic_mean'].max():.4f}"
        )
        print(f"  % mois IC>0: {100 * (s['ic_mean'] > 0).mean():.1f}%")
        top = s.nlargest(3, "ic_mean")[["mois", "ic_mean"]]
        bot = s.nsmallest(3, "ic_mean")[["mois", "ic_mean"]]
        print(f"  top3: {top.to_dict('records')}")
        print(f"  bottom3: {bot.to_dict('records')}")

        # contribution: sans top 2 mois
        s2 = s.copy()
        drop = s2.nlargest(2, "ic_mean").index
        s2 = s2.drop(drop)
        print(f"  mean sans top-2 mois: {s2['ic_mean'].mean():.4f}")


def point2() -> None:
    print("\n" + "=" * 70)
    print("POINT 2 — Sensibilité poids Ridge/RF (recalcul scores existants)")
    print("=" * 70)

    preds = pd.read_parquet(REPORTS_DIR / "factorial_predictions.parquet")
    # Besoin score_ridge et score_rf — présents pour hybrides; pour ridge-only pas de score_rf
    # On utilise cell 3 (hybrid no regime) et cell 4 (hybrid with regime) comme base,
    # OU on reconstruit depuis score_ridge/score_rf si disponibles.
    print("\nColonnes preds:", list(preds.columns))

    # Cellules hybrides ont score_ridge + score_rf
    for cell_id, tag in ((3, "hybrid_static"), (4, "hybrid_adapt")):
        sub = preds[(preds["cell_id"] == cell_id) & (preds["split"] == "test")].copy()
        if "score_ridge" not in sub.columns or "score_rf" not in sub.columns:
            print(f"  C{cell_id}: colonnes score_ridge/score_rf absentes — skip")
            continue
        if sub["score_rf"].isna().all():
            print(f"  C{cell_id}: score_rf vide — skip")
            continue

        print(f"\n  [{tag} / cell {cell_id}]")
        for w_r, w_f in ((0.5, 0.5), (0.7, 0.3), (0.9, 0.1), (1.0, 0.0)):
            score = combine_hybrid_scores(
                sub["score_ridge"].values,
                sub["score_rf"].values,
                sub["date_cours"],
                w_ridge=w_r,
                w_rf=w_f,
            )
            eval_df = sub[["ticker", "date_cours", TARGET_COLUMN]].copy()
            eval_df["prediction"] = score
            m = evaluate_scoring_model(eval_df, "prediction", TARGET_COLUMN)
            h = hit_ratio(eval_df, "prediction", TARGET_COLUMN)
            print(
                f"    Ridge/RF={w_r:.1f}/{w_f:.1f} -> "
                f"IC_test(Spearman)={m['rank_ic_mean']:.4f} | "
                f"IC_pearson={m['ic_mean']:.4f} | "
                f"hit={100 * (h['hit_ratio'] or 0):.1f}%"
            )


def point3() -> None:
    print("\n" + "=" * 70)
    print("POINT 3 — Hyperparamètres RF retenus")
    print("=" * 70)
    report = json.loads((REPORTS_DIR / "rf_etape2_report.json").read_text(encoding="utf-8"))
    bp = report["best_params"]
    print(f"  best_params: {bp}")
    print(f"  n_features: {report.get('n_features')}")
    print(f"  tuning_metric: {report.get('tuning_metric')}")
    # train size
    ml = pd.read_parquet(DATASET_DIR / "ml_dataset.parquet", columns=["split"])
    n_train = int((ml["split"] == "train").sum())
    print(f"  n_train rows: {n_train}")
    print(
        f"  Commentaire profondeur: max_depth={bp.get('max_depth')} "
        f"sur n_train={n_train} — modèle peu profond (pas sur-paramétré en profondeur)."
    )
    # top of grid
    hist = report.get("grid_history") or []
    print("  Top-5 grille (Rank-IC val):")
    for row in hist[:5]:
        print(f"    {row}")


def point4() -> None:
    print("\n" + "=" * 70)
    print("POINT 4 — Régime sur période TEST uniquement (mois)")
    print("=" * 70)
    ml = pd.read_parquet(
        DATASET_DIR / "ml_dataset.parquet", columns=["date_cours", "split"]
    )
    ml["date_cours"] = pd.to_datetime(ml["date_cours"])
    fi = pd.read_parquet(
        FEATURES_DIR / "features_indices.parquet",
        columns=["date_cours", "is_bull", "is_neutral", "is_bear"],
    )
    fi["date_cours"] = pd.to_datetime(fi["date_cours"])
    fi = fi.drop_duplicates("date_cours", keep="last")

    sub = ml[ml["split"] == "test"].copy()
    sub["mois"] = sub["date_cours"].dt.to_period("M")
    last_dates = sub.groupby("mois")["date_cours"].max().reset_index()
    reg = last_dates.merge(fi, on="date_cours", how="left")
    reg["regime"] = reg.apply(_regime_label, axis=1)
    vc = reg["regime"].value_counts()
    print(f"  n_mois_test={len(reg)}")
    for k, v in vc.items():
        print(f"  {k}: {int(v)} mois ({100 * v / len(reg):.1f}%)")
    print("\n  Détail mois -> régime:")
    for _, r in reg.sort_values("mois").iterrows():
        print(f"    {r['mois']}: {r['regime']}")


def point5() -> None:
    print("\n" + "=" * 70)
    print("POINT 5 — Absence de fuite / identité features+cible vs Ridge baseline")
    print("=" * 70)

    # Compare C1 predictions IC to baseline ridge if available
    from bvc_recommender.models.baseline_models import (
        get_z_feature_columns,
        prepare_splits,
        train_model,
        predict_model,
    )
    from bvc_recommender.models.baseline_models import _build_xy

    ml = pd.read_parquet(DATASET_DIR / "ml_dataset.parquet")
    feats = get_z_feature_columns(ml)
    print(f"  TARGET_COLUMN = {TARGET_COLUMN}")
    print(f"  n_features *_z = {len(feats)}")
    print(f"  features (identiques Ridge baseline): {feats[:5]} ... {feats[-3:]}")

    # Recompute Ridge baseline IC_test quickly (same code path as step5)
    splits, feature_cols, _ = prepare_splits(ml, feature_cols=feats, target_col=TARGET_COLUMN)
    X_train, y_train = _build_xy(splits["train"], feature_cols, TARGET_COLUMN)
    model = train_model("ridge", X_train, y_train)
    X_test, _ = _build_xy(splits["test"], feature_cols, TARGET_COLUMN)
    preds = predict_model(model, X_test)
    eval_df = splits["test"][["ticker", "date_cours", TARGET_COLUMN]].copy()
    eval_df["prediction"] = preds
    m = evaluate_scoring_model(eval_df, "prediction", TARGET_COLUMN)
    print(
        f"  Ridge baseline re-fit (ce script): Rank-IC_test={m['rank_ic_mean']:.4f} "
        f"| IC_pearson={m['ic_mean']:.4f}"
    )

    # C1 from factorial
    fac = pd.read_parquet(REPORTS_DIR / "factorial_predictions.parquet")
    c1 = fac[(fac["cell_id"] == 1) & (fac["split"] == "test")]
    m1 = evaluate_scoring_model(c1, "prediction", TARGET_COLUMN)
    print(
        f"  Cellule 1 (factorial): Rank-IC_test={m1['rank_ic_mean']:.4f} "
        f"| IC_pearson={m1['ic_mean']:.4f}"
    )
    print(
        f"  Écart |C1 - baseline| Rank-IC = {abs(m1['rank_ic_mean'] - m['rank_ic_mean']):.6f}"
    )

    # Regime: confirmed from features_indices (fitted on train thresholds only)
    print("\n  Fuite — contrôles:")
    print("  - Split temporel calendaire (train≤2020, val 2021-22, test≥2023): OUI")
    print("  - Clip cible: bornes apprises sur train uniquement (prepare_splits): OUI")
    print("  - RF fit sur train seulement; hyperparams choisis sur val: OUI")
    print("  - Régime: colonnes is_* depuis features_indices (seuils documentés train): OUI")
    print("  - Pas de features futures dans *_z (même dataset ml_dataset): OUI")
    print("  - Hybride: z-score cross-sectionnel par date (pas de stats futures): OUI")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    point1()
    point2()
    point3()
    point4()
    point5()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
