"""
Diagnostic lecture seule — verification outliers return_dispersion apres correction splits.

Recalcule UNIQUEMENT la dispersion cross-sectionnelle depuis market_data_cours_historique
(prix corriges en base), compare aux stats AVANT (features_indices local), et ecrit
diagnostics/dispersion_before_after.png.

Usage :
  py -3 diagnostics/dispersion_split_correction_check.py
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
    DATA_PROCESSED_DIR,
    FEATURES_DIR,
    TABLE_COURS_HISTORIQUE,
    load_env_file,
)
from bvc_recommender.features._utils import find_col  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "dispersion_before_after.png"
OUT_TOP20 = OUT_DIR / "dispersion_top20_after.csv"

START = "2015-01-01"
END = "2025-12-31"
JUMP_THRESH = 0.30  # |ret| > 30% = saut suspect

# Stats AVANT connues (diagnostic regime_variability_check, avant correction splits)
# Utilisees si le snapshot local features_indices a deja ete ecrase.
BEFORE_FALLBACK_STATS = {
    "n": 2703,
    "median": 0.018994,
    "mean": 0.049084,
    "std": 0.250976,
    "p90": 0.028414,
    "p95": 0.049698,
    "p99": 1.085015,
    "p999": 4.614872,
    "max": 4.705627,
    "max_date": "2018-09-27",
    "n_gt_10x_median": None,  # calcule si serie avant disponible
}


def _load_before_dispersion() -> tuple[pd.Series, str]:
    """Serie return_dispersion AVANT (snapshot features_indices local)."""
    path = FEATURES_DIR / "features_indices.parquet"
    if not path.is_file():
        return pd.Series(dtype=float), "fallback:stats_connues"
    df = pd.read_parquet(path)
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    df["return_dispersion"] = pd.to_numeric(df["return_dispersion"], errors="coerce")
    df = df.dropna(subset=["date_cours", "return_dispersion"])
    df = df[(df["date_cours"] >= START) & (df["date_cours"] <= END)]
    df = df.drop_duplicates("date_cours", keep="last").sort_values("date_cours")
    s = df.set_index("date_cours")["return_dispersion"]
    # Si le max a deja chute, le snapshot local a probablement ete regenere apres correction
    if float(s.max()) < 1.0:
        return s, f"local:{path} (ATTENTION: max deja <1 — snapshot peut-etre deja corrige)"
    return s, f"local:{path}"


def _load_cours_corrected() -> tuple[pd.DataFrame, str]:
    """Charge les prix corriges : Postgres > REST (page_size=1000) > parquet local."""
    load_env_file()

    from bvc_recommender.data.loader import (
        fetch_table,
        fetch_table_postgres,
        get_supabase_client,
    )

    pg = fetch_table_postgres(TABLE_COURS_HISTORIQUE)
    if pg is not None and not pg.empty:
        return pg, f"postgres:{TABLE_COURS_HISTORIQUE}"

    # Cache REST pour eviter de re-telecharger ~240k lignes a chaque run
    cache_path = OUT_DIR / "_cache_cours_rest.parquet"
    try:
        client = get_supabase_client()
        # PostgREST plafonne souvent a 1000 lignes/requete : page_size doit etre <= 1000
        # sinon fetch_table s'arrete apres la 1re page (len(batch) < limit).
        print("  telechargement REST market_data_cours_historique (pagination)...")
        df = fetch_table(
            client,
            TABLE_COURS_HISTORIQUE,
            columns="id,ticker,date_cours,prix_cloture,prix_courant",
            order_by="id",
            page_size=1000,
            checkpoint_path=OUT_DIR / "_checkpoints" / "cours_rest.parquet",
        )
        if df is not None and not df.empty:
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
            return df, f"supabase_rest:{TABLE_COURS_HISTORIQUE} (n={len(df)})"
    except Exception as exc:
        print(f"[warn] REST Supabase echec : {exc}")
        if cache_path.is_file():
            print(f"  fallback cache REST : {cache_path}")
            return pd.read_parquet(cache_path), f"cache_rest:{cache_path}"

    local = DATA_PROCESSED_DIR / "market_data_cours_historique.parquet"
    if local.is_file():
        return (
            pd.read_parquet(local),
            f"local:{local} (ATTENTION: peut etre non mis a jour vs DB)",
        )

    raise FileNotFoundError(
        "Impossible de charger market_data_cours_historique (DB + local absents)."
    )


def _prepare_cours(cours: pd.DataFrame) -> pd.DataFrame:
    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
    tick_col = find_col(c, "ticker", "code") or "ticker"
    c["date_cours"] = pd.to_datetime(c[date_col], errors="coerce")
    c["prix_cloture"] = pd.to_numeric(c[price_col], errors="coerce")
    c["ticker"] = c[tick_col].astype(str).str.strip()
    c = c.dropna(subset=["date_cours", "prix_cloture", "ticker"])
    c = c[(c["date_cours"] >= START) & (c["date_cours"] <= END)]
    c = c.sort_values(["ticker", "date_cours"]).drop_duplicates(
        ["ticker", "date_cours"], keep="last"
    )
    return c[["ticker", "date_cours", "prix_cloture"]].reset_index(drop=True)


def _compute_dispersion_and_returns(
    daily: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Recalcule UNIQUEMENT return_dispersion + panel des rendements journaliers.

    Returns
    -------
    dispersion : Series indexee par date_cours
    rets : DataFrame ticker, date_cours, prix_cloture, daily_ret, prev_price
    """
    parts: list[pd.DataFrame] = []
    for ticker, grp in daily.groupby("ticker", sort=False):
        g = grp.sort_values("date_cours").copy()
        g["prev_price"] = g["prix_cloture"].shift(1)
        g["daily_ret"] = g["prix_cloture"].pct_change(fill_method=None)
        parts.append(g)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.dropna(subset=["daily_ret"])

    disp = (
        panel.groupby("date_cours", sort=True)["daily_ret"]
        .agg(lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan)
        .rename("return_dispersion")
        .dropna()
    )
    return disp, panel


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    med = float(s.median())
    mx = float(s.max())
    if isinstance(s.index, pd.DatetimeIndex):
        max_date = str(s.idxmax().date())
    else:
        max_date = str(s.idxmax())
    return {
        "n": int(len(s)),
        "median": med,
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "p999": float(s.quantile(0.999)),
        "max": mx,
        "max_date": max_date,
        "n_gt_10x_median": int((s > 10 * med).sum()) if med > 0 else 0,
    }


def _print_comparison(before: dict, after: dict, before_src: str, after_src: str) -> None:
    print("\n" + "=" * 78)
    print("COMPARAISON STATISTIQUES return_dispersion (2015-2025)")
    print("=" * 78)
    print(f"AVANT source : {before_src}")
    print(f"APRES source : {after_src}")
    print()
    headers = ("metrique", "AVANT", "APRES", "delta")
    rows = [
        ("n jours", before.get("n"), after.get("n"), None),
        ("mediane", before.get("median"), after.get("median"), True),
        ("moyenne", before.get("mean"), after.get("mean"), True),
        ("ecart-type", before.get("std"), after.get("std"), True),
        ("p90", before.get("p90"), after.get("p90"), True),
        ("p95", before.get("p95"), after.get("p95"), True),
        ("p99", before.get("p99"), after.get("p99"), True),
        ("p99.9", before.get("p999"), after.get("p999"), True),
        ("max", before.get("max"), after.get("max"), True),
        ("date du max", before.get("max_date"), after.get("max_date"), False),
        (
            "n > 10x mediane",
            before.get("n_gt_10x_median"),
            after.get("n_gt_10x_median"),
            False,
        ),
    ]
    print(f"{'metrique':<18} {'AVANT':>14} {'APRES':>14} {'delta':>14}")
    print("-" * 62)
    for name, b, a, is_num in rows:
        if b is None and a is None:
            continue
        if is_num and isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            print(f"{name:<18} {b:14.6f} {a:14.6f} {delta:+14.6f}")
        elif name == "n jours" or name == "n > 10x mediane":
            bs = "—" if b is None else str(int(b))
            as_ = "—" if a is None else str(int(a))
            if b is not None and a is not None:
                d = f"{int(a) - int(b):+d}"
            else:
                d = "—"
            print(f"{name:<18} {bs:>14} {as_:>14} {d:>14}")
        else:
            print(f"{name:<18} {str(b):>14} {str(a):>14} {'':>14}")


def _top_contributors(panel: pd.DataFrame, date: pd.Timestamp, k: int = 3) -> list[dict]:
    day = panel.loc[panel["date_cours"] == date].copy()
    if day.empty:
        return []
    med = float(day["daily_ret"].median())
    day["abs_dev"] = (day["daily_ret"] - med).abs()
    top = day.nlargest(k, "abs_dev")
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "ticker": r["ticker"],
                "daily_ret": float(r["daily_ret"]),
                "abs_dev": float(r["abs_dev"]),
                "prix": float(r["prix_cloture"]),
                "prev_price": float(r["prev_price"]) if pd.notna(r["prev_price"]) else np.nan,
            }
        )
    return rows


def _check_jumps(contribs: list[dict]) -> list[dict]:
    suspects = []
    for c in contribs:
        ret = c["daily_ret"]
        if abs(ret) > JUMP_THRESH:
            suspects.append(c)
    return suspects


def _plot_hist(before: pd.Series, after: pd.Series, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle(
        "return_dispersion — avant vs apres correction splits (2015-2025)",
        fontsize=12,
        fontweight="bold",
    )

    b = before.dropna().to_numpy(dtype=float)
    a = after.dropna().to_numpy(dtype=float)

    # Couper l'axe x apres au p99.5 apres + marge, pour lisibilite
    x_max_after = float(np.quantile(a, 0.995)) * 1.5 if len(a) else 0.1
    x_max_after = max(x_max_after, float(np.median(a)) * 20 if len(a) else 0.1)

    axes[0].hist(b, bins=80, color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].axvline(np.median(b), color="black", linestyle="--", label=f"mediane={np.median(b):.4f}")
    axes[0].axvline(np.max(b), color="#1f77b4", linestyle=":", label=f"max={np.max(b):.3f}")
    axes[0].set_title(f"AVANT (n={len(b)})")
    axes[0].set_xlabel("return_dispersion")
    axes[0].set_ylabel("Nombre de jours")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        a,
        bins=80,
        color="#2ca02c",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        range=(0, max(x_max_after, float(np.max(a)) * 1.05)),
    )
    axes[1].axvline(np.median(a), color="black", linestyle="--", label=f"mediane={np.median(a):.4f}")
    axes[1].axvline(np.max(a), color="#1f77b4", linestyle=":", label=f"max={np.max(a):.4f}")
    axes[1].set_title(f"APRES (n={len(a)})")
    axes[1].set_xlabel("return_dispersion")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Inset log-scale note if before has huge outliers
    if np.max(b) > 1:
        axes[0].text(
            0.98,
            0.95,
            f"max={np.max(b):.2f} (hors echelle utile)",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#d62728",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure ecrite : {out_path}")


def _verdict(after: dict, residual_suspects: list[dict]) -> str:
    """
    Correction suffisante si max du meme ordre de grandeur que p99
    (pas d'ecart disproportionne type max >> p99 comme avant).
    """
    mx = after["max"]
    p99 = after["p99"]
    med = after["median"]
    # Avant: max/p99 ~ 4.3, max/mediane ~ 248
    ratio_max_p99 = mx / p99 if p99 > 0 else float("inf")
    ratio_max_med = mx / med if med > 0 else float("inf")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  max / p99     = {ratio_max_p99:.2f}  (avant ~4.3)")
    print(f"  max / mediane = {ratio_max_med:.1f}  (avant ~248)")
    print(f"  n > 10x med   = {after['n_gt_10x_median']}")
    print(f"  sauts >30% sur top-20 contributeurs : {len(residual_suspects)}")

    # Criteres: max proche de p99 (ratio < 3) ET max < 0.5 (ordre de grandeur market)
    # ET peu de sauts residuels sur les top contributeurs
    ok_scale = ratio_max_p99 < 3.0 and mx < 0.50
    ok_jumps = len(residual_suspects) == 0

    if ok_scale and ok_jumps:
        print("\n>>> Correction suffisante")
        print(
            "    Le maximum apres correction est coherent avec la distribution "
            "(plus d'ecart disproportionne type 4.7)."
        )
        return "Correction suffisante"

    print("\n>>> Anomalies residuelles detectees")
    if not ok_scale:
        print(
            f"    Echelle encore suspecte : max={mx:.4f}, p99={p99:.4f}, "
            f"mediane={med:.4f}."
        )
    if residual_suspects:
        print("    Tickers/dates a investiguer (|ret| > 30%) :")
        seen = set()
        for s in residual_suspects:
            key = (s["date"], s["ticker"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"      - {s['date']} | {s['ticker']}: ret={s['daily_ret']:+.1%} "
                f"(prix {s.get('prev_price', float('nan')):.4f} -> {s['prix']:.4f})"
            )
    return "Anomalies residuelles detectees"


def main() -> int:
    print("=" * 78)
    print("DIAGNOSTIC — return_dispersion avant/apres correction splits")
    print("=" * 78)

    before_s, before_src = _load_before_dispersion()
    if before_s.empty or float(before_s.max()) < 1.0:
        # Snapshot local deja corrige ou absent : utiliser stats connues pour AVANT
        before_stats = dict(BEFORE_FALLBACK_STATS)
        # Pour l'histogramme AVANT : si serie dispo mais max<1, on ne peut pas
        # reconstruire les outliers — on genere une serie synthetique indicative
        # uniquement pour le panneau gauche? Mieux: charger stats et plotter
        # l'ancienne serie si max>=1, sinon noter dans le plot.
        if not before_s.empty and float(before_s.max()) < 1.0:
            print(
                f"\n[info] Snapshot features_indices max={float(before_s.max()):.4f} "
                "< 1 — stats AVANT = valeurs du diagnostic precedent (connues)."
            )
            before_for_plot = None
            before_src = "stats diagnostique precedent (features_indices deja maj?)"
        elif before_s.empty:
            before_for_plot = None
            before_src = "stats diagnostique precedent (parquet absent)"
        else:
            before_stats = _stats(before_s)
            before_for_plot = before_s
    else:
        before_stats = _stats(before_s)
        before_for_plot = before_s
        # Completer n_gt_10x si fallback avait None
        if before_stats.get("n_gt_10x_median") is None:
            med = before_stats["median"]
            before_stats["n_gt_10x_median"] = int((before_s > 10 * med).sum())

    if before_stats.get("n_gt_10x_median") is None and before_for_plot is not None:
        med = before_stats["median"]
        before_stats["n_gt_10x_median"] = int((before_for_plot > 10 * med).sum())

    print("\nChargement des prix corriges...")
    cours_raw, after_src = _load_cours_corrected()
    print(f"  source prix : {after_src}  ({len(cours_raw)} lignes brutes)")
    daily = _prepare_cours(cours_raw)
    print(f"  apres prep  : {len(daily)} lignes | {daily['ticker'].nunique()} tickers")
    print(f"  periode     : {daily['date_cours'].min().date()} -> {daily['date_cours'].max().date()}")

    if daily.empty:
        print("ERREUR : aucune ligne de cours apres preparation (colonnes/dates?).")
        return 1

    print("\nRecalcul UNIQUEMENT de return_dispersion...")
    after_s, panel = _compute_dispersion_and_returns(daily)
    if after_s.empty:
        print("ERREUR : dispersion vide apres recalcul.")
        return 1
    after_stats = _stats(after_s)
    print(f"  jours avec dispersion : {len(after_s)}")

    _print_comparison(before_stats, after_stats, before_src, after_src)

    # Top 20 dates
    print("\n" + "=" * 78)
    print("TOP 20 DATES — dispersion la plus elevee APRES correction")
    print("=" * 78)
    top20 = after_s.nlargest(20)
    residual_suspects: list[dict] = []
    export_rows: list[dict] = []

    for i, (dt, disp) in enumerate(top20.items(), start=1):
        contribs = _top_contributors(panel, dt, k=3)
        jumps = _check_jumps(contribs)
        print(f"\n{i:2d}. {dt.date()}  dispersion={disp:.6f}")
        for j, c in enumerate(contribs, start=1):
            flag = "  << SAUT >30%" if abs(c["daily_ret"]) > JUMP_THRESH else ""
            print(
                f"      #{j} {c['ticker']:<12} ret={c['daily_ret']:+.2%}  "
                f"|dev_med|={c['abs_dev']:.4f}  "
                f"prix={c['prev_price']:.4f}->{c['prix']:.4f}{flag}"
            )
            export_rows.append(
                {
                    "rank_date": i,
                    "date_cours": dt.date().isoformat(),
                    "return_dispersion": disp,
                    "rank_ticker": j,
                    "ticker": c["ticker"],
                    "daily_ret": c["daily_ret"],
                    "abs_dev_from_median": c["abs_dev"],
                    "prev_price": c["prev_price"],
                    "prix_cloture": c["prix"],
                    "jump_gt_30pct": abs(c["daily_ret"]) > JUMP_THRESH,
                }
            )
        for j in jumps:
            residual_suspects.append(
                {
                    "date": dt.date().isoformat(),
                    "ticker": j["ticker"],
                    "daily_ret": j["daily_ret"],
                    "prix": j["prix"],
                    "prev_price": j["prev_price"],
                }
            )

    pd.DataFrame(export_rows).to_csv(OUT_TOP20, index=False, float_format="%.8f")
    print(f"\nExport top-20 : {OUT_TOP20}")

    # Histogramme
    print("\n" + "=" * 78)
    print("HISTOGRAMME COMPARATIF")
    print("=" * 78)
    if before_for_plot is None:
        # Reconstituer un proxy AVANT pour le plot : serie apres + reinjection
        # des stats connues via message; utiliser features si dispo meme corrige
        # pour forme, et annoter. Mieux: plotter stats textuelles a gauche.
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(
            "return_dispersion — avant vs apres correction splits (2015-2025)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].axis("off")
        txt = (
            "AVANT (stats diagnostique precedent)\n\n"
            f"n      = {before_stats['n']}\n"
            f"mediane= {before_stats['median']:.6f}\n"
            f"moyenne= {before_stats['mean']:.6f}\n"
            f"std    = {before_stats['std']:.6f}\n"
            f"p90    = {before_stats['p90']:.6f}\n"
            f"p95    = {before_stats['p95']:.6f}\n"
            f"p99    = {before_stats['p99']:.6f}\n"
            f"p99.9  = {before_stats['p999']:.6f}\n"
            f"max    = {before_stats['max']:.6f}\n"
            f"date max = {before_stats['max_date']}\n\n"
            "(serie journaliere avant indisponible:\n"
            " snapshot local deja corrige ou absent)"
        )
        axes[0].text(0.05, 0.95, txt, va="top", family="monospace", fontsize=9)
        axes[0].set_title("AVANT")

        a = after_s.dropna().to_numpy(dtype=float)
        axes[1].hist(a, bins=80, color="#2ca02c", alpha=0.85, edgecolor="white", linewidth=0.3)
        axes[1].axvline(np.median(a), color="black", linestyle="--", label=f"mediane={np.median(a):.4f}")
        axes[1].axvline(np.max(a), color="#1f77b4", linestyle=":", label=f"max={np.max(a):.4f}")
        axes[1].set_title(f"APRES (n={len(a)})")
        axes[1].set_xlabel("return_dispersion")
        axes[1].set_ylabel("Nombre de jours")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure ecrite : {OUT_PNG}")
    else:
        if before_stats.get("n_gt_10x_median") is None:
            before_stats["n_gt_10x_median"] = int(
                (before_for_plot > 10 * before_stats["median"]).sum()
            )
        _plot_hist(before_for_plot, after_s, OUT_PNG)

    _verdict(after_stats, residual_suspects)
    print("\nDiagnostic termine (lecture seule — aucune ecriture en base).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
