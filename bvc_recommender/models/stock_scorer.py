"""
Sous-module B — Scoring des actions via Temporal Fusion Transformer (TFT).

Originalité : injection du régime de marché (Composante A) comme variables
connues à l'avance (time_varying_known_reals), afin que l'attention s'adapte
au régime.

Production : one-hot discret [is_bull, is_neutral, is_bear] (règle à seuils).
Modularité :
- ``regime_encoding`` : ``"onehot"`` (défaut) ou ``"ordinal"`` (regime_score ∈ {-1,0,+1})
- ``target_type`` : ``"alpha_ajuste_risque"`` (défaut) ou ``"rang_cross_sectionnel"``
- ``sequence_length`` / ``max_encoder_length`` : fenêtre encodeur (défaut 20)
- ``known_reals`` / ``attach_regime_features`` acceptent aussi un vecteur continu (ex. mu_*)

Validation temporelle :
- train 2015-2020 | val 2021-2022 | test 2023-2025

Sorties : quantiles Q10 (pessimiste) / Q50 (central) / Q90 (optimiste)
de ``alpha_ajuste_risque``.

Traçabilité données : pipeline post-correction des splits d'actions BVC
(prix corrigés dans market_data_cours_historique, 2026-07-28). Ne pas
réutiliser de checkpoints / parquet TFT antérieurs à cette date.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import (
    RANK_TARGET_COLUMN,
    TARGET_COLUMN,
    add_cross_sectional_rank_target,
)
from bvc_recommender.models.baseline_models import prepare_splits
from bvc_recommender.models.metrics import evaluate_scoring_model
from bvc_recommender.models.regime_detector import REGIME_COLUMNS

logger = logging.getLogger(__name__)

# Provenance méthodologie article — post-correction splits (2026-07-28)
DATA_VERSION_NOTE = "post_split_correction_2026-07-28"

TFT_QUANTILES = [0.1, 0.5, 0.9]
DEFAULT_ENCODER_LENGTH = 20
DEFAULT_MAX_TRAIN_ROWS = 40_000
DEFAULT_HIDDEN_SIZES = (2, 4, 8, 16, 32, 64)
DEFAULT_DROPOUTS = (0.1, 0.2, 0.3, 0.4, 0.5)
REGIME_SCORE_COLUMN = "regime_score"
REGIME_ENCODING_ONEHOT = "onehot"
REGIME_ENCODING_ORDINAL = "ordinal"
TARGET_TYPE_ALPHA = "alpha_ajuste_risque"
TARGET_TYPE_RANK = "rang_cross_sectionnel"
_SPLIT_ORDER = ("train", "val", "test")

# Défaut one-hot neutre si régime manquant (discret).
_DEFAULT_DISCRETE_FILL = {"is_bull": 0.0, "is_neutral": 1.0, "is_bear": 0.0}


def resolve_target_column(target_type: str) -> str:
    """Mappe ``target_type`` → nom de colonne d'entraînement."""
    if target_type in (TARGET_TYPE_ALPHA, TARGET_COLUMN):
        return TARGET_COLUMN
    if target_type in (TARGET_TYPE_RANK, RANK_TARGET_COLUMN):
        return RANK_TARGET_COLUMN
    raise ValueError(
        f"target_type inconnu: {target_type!r}. "
        f"Attendu: {TARGET_TYPE_ALPHA!r} ou {TARGET_TYPE_RANK!r}."
    )


def apply_regime_encoding(
    df: pd.DataFrame,
    regime_encoding: str = REGIME_ENCODING_ONEHOT,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Sélectionne les known_reals selon l'encodage.

    - ``onehot`` : is_bull / is_neutral / is_bear
    - ``ordinal`` : regime_score = -1 (bear), 0 (neutral), +1 (bull)
    """
    encoding = (regime_encoding or REGIME_ENCODING_ONEHOT).strip().lower()
    out = df.copy()
    if encoding == REGIME_ENCODING_ONEHOT:
        return out, list(REGIME_COLUMNS)
    if encoding == REGIME_ENCODING_ORDINAL:
        for col in REGIME_COLUMNS:
            if col not in out.columns:
                out[col] = _DEFAULT_DISCRETE_FILL.get(col, 0.0)
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
                _DEFAULT_DISCRETE_FILL.get(col, 0.0)
            )
        bull = out["is_bull"].to_numpy(dtype=float)
        bear = out["is_bear"].to_numpy(dtype=float)
        out[REGIME_SCORE_COLUMN] = np.where(
            bull >= 0.5, 1.0, np.where(bear >= 0.5, -1.0, 0.0)
        ).astype(float)
        return out, [REGIME_SCORE_COLUMN]
    raise ValueError(
        f"regime_encoding inconnu: {regime_encoding!r}. "
        f"Attendu: {REGIME_ENCODING_ONEHOT!r} ou {REGIME_ENCODING_ORDINAL!r}."
    )


def prepare_tft_panel(
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    regime_encoding: str = REGIME_ENCODING_ONEHOT,
    target_type: str = TARGET_TYPE_ALPHA,
    known_reals: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Joint le régime, applique l'encodage, et matérialise la cible d'entraînement.

    Les métriques reportées restent toujours vs ``alpha_ajuste_risque`` pour
    comparabilité Ridge / LightGBM / TFT baseline.
    """
    # Toujours attacher le one-hot (source de vérité), puis dériver l'ordinal si besoin.
    enriched = attach_regime_features(df, regime_source, columns=list(REGIME_COLUMNS))
    if known_reals is not None:
        known = list(known_reals)
        if REGIME_SCORE_COLUMN in known and REGIME_SCORE_COLUMN not in enriched.columns:
            enriched, _ = apply_regime_encoding(enriched, REGIME_ENCODING_ORDINAL)
    else:
        enriched, known = apply_regime_encoding(enriched, regime_encoding)

    target_col = resolve_target_column(target_type)
    if target_col == RANK_TARGET_COLUMN:
        enriched = add_cross_sectional_rank_target(
            enriched, source_col=TARGET_COLUMN, out_col=RANK_TARGET_COLUMN
        )
    return enriched, known, target_col


def _hidden_continuous_size(hidden_size: int, override: int | None = None) -> int:
    """Adapte hidden_continuous_size aux petits hidden_size (2/4/8)."""
    if override is not None:
        return int(override)
    if hidden_size < 16:
        return max(1, hidden_size // 2)
    return max(8, hidden_size // 2)


def attach_regime_features(
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Joint les colonnes de régime au panel titres.

    Accepte :
    - panel journalier (``date_cours`` + colonnes) — ex. features_indices
    - historique mensuel (``year`` / ``month``) — ex. résumé step4 / fuzzy ablation

    ``columns`` / ``fill_values`` permettent un conditionnement continu (mu_*)
    sans réécrire le TFT.
    """
    cols = list(columns) if columns is not None else list(REGIME_COLUMNS)
    fills = fill_values
    if fills is None:
        if set(cols) <= set(REGIME_COLUMNS):
            fills = dict(_DEFAULT_DISCRETE_FILL)
        else:
            fills = {c: 1.0 / max(len(cols), 1) for c in cols}

    out = df.copy()
    out["date_cours"] = pd.to_datetime(out["date_cours"], errors="coerce")
    regime = regime_source.copy()
    # Rétrocompat HMM : is_sideways → is_neutral
    if "is_neutral" in cols and "is_neutral" not in regime.columns and "is_sideways" in regime.columns:
        regime["is_neutral"] = regime["is_sideways"]

    if "date_cours" in regime.columns:
        regime["date_cours"] = pd.to_datetime(regime["date_cours"], errors="coerce")
        keep = ["date_cours", *[c for c in cols if c in regime.columns]]
        regime = regime[keep].dropna(subset=["date_cours"]).sort_values("date_cours")
        regime = regime.drop_duplicates("date_cours", keep="last")
        out = out.sort_values("date_cours")
        out = pd.merge_asof(
            out,
            regime,
            on="date_cours",
            direction="backward",
        )
        if "ticker" in out.columns:
            out = out.sort_values(["ticker", "date_cours"])
    elif {"year", "month"}.issubset(regime.columns):
        out["year"] = out["date_cours"].dt.year
        out["month"] = out["date_cours"].dt.month
        keep = ["year", "month", *[c for c in cols if c in regime.columns]]
        regime = regime[keep]
        out = out.merge(regime, on=["year", "month"], how="left")
        out = out.sort_values(["ticker", "date_cours"])
        for col in cols:
            if col in out.columns:
                out[col] = out.groupby("ticker")[col].ffill()
        out = out.drop(columns=["year", "month"], errors="ignore")
    else:
        raise ValueError(
            "regime_source doit contenir date_cours (journalier) ou year+month (mensuel)."
        )

    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        default = fills.get(col, 0.0)
        out[col] = out[col].fillna(default)

    # Garantir un one-hot valide pour le régime discret de production
    if set(cols) == set(REGIME_COLUMNS):
        s = out[list(REGIME_COLUMNS)].sum(axis=1)
        mask = s <= 0
        out.loc[mask, "is_neutral"] = 1.0
        out.loc[mask, "is_bull"] = 0.0
        out.loc[mask, "is_bear"] = 0.0

    return out


def _prepare_tft_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    known_reals: Sequence[str] | None = None,
    target_col: str = TARGET_COLUMN,
    subsample: int | None = None,
) -> pd.DataFrame:
    known = list(known_reals) if known_reals is not None else list(REGIME_COLUMNS)
    frame = df.sort_values(["ticker", "date_cours"]).copy()
    for col in feature_cols + known + [target_col]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[col] = frame[col].fillna(0.0)

    frame["time_idx"] = frame.groupby("ticker").cumcount().astype(int)

    if subsample and len(frame) > subsample:
        n_tickers = max(1, frame["ticker"].nunique())
        per_ticker = max(30, subsample // n_tickers)
        parts: list[pd.DataFrame] = []
        for _, grp in frame.groupby("ticker", sort=False):
            if len(grp) > per_ticker:
                step = max(1, len(grp) // per_ticker)
                parts.append(grp.iloc[::step])
            else:
                parts.append(grp)
        frame = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date_cours"])
        frame["time_idx"] = frame.groupby("ticker").cumcount().astype(int)

    return frame


def _build_timeseries_dataset(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    known_reals: Sequence[str] | None = None,
    target_col: str = TARGET_COLUMN,
    max_encoder_length: int = DEFAULT_ENCODER_LENGTH,
):
    """Construit le TimeSeriesDataSet ; ``known_reals`` reste injectable (discret ou continu)."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    known = list(known_reals) if known_reals is not None else list(REGIME_COLUMNS)
    return TimeSeriesDataSet(
        frame,
        time_idx="time_idx",
        target=target_col,
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_reals=known,
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        randomize_length=None,
    )


def train_tft_model(
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    known_reals: Sequence[str] | None = None,
    regime_encoding: str = REGIME_ENCODING_ONEHOT,
    target_type: str = TARGET_TYPE_ALPHA,
    target_col: str | None = None,
    max_encoder_length: int = DEFAULT_ENCODER_LENGTH,
    sequence_length: int | None = None,
    max_epochs: int = 4,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    batch_size: int = 256,
    learning_rate: float = 0.03,
    hidden_size: int = 16,
    attention_head_size: int = 2,
    dropout: float = 0.1,
    hidden_continuous_size: int | None = None,
    early_stopping_patience: int = 2,
    save_model: bool = True,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Entraîne un TFT avec injection du régime et retourne modèle + datasets + métadonnées.

    ``sequence_length`` est un alias de ``max_encoder_length`` (fenêtre d'historique).
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    class _EpochLossHistory(Callback):
        """Capture train/val loss (QuantileLoss) à chaque epoch."""

        def __init__(self) -> None:
            super().__init__()
            self.history: list[dict[str, Any]] = []

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            metrics = trainer.callback_metrics
            train_loss = _metric_float(
                metrics.get("train_loss")
                or metrics.get("train_loss_epoch")
                or metrics.get("loss")
            )
            val_loss = _metric_float(
                metrics.get("val_loss") or metrics.get("val_loss_epoch")
            )
            row = {
                "epoch": int(trainer.current_epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_quantile_loss": train_loss,
                "val_quantile_loss": val_loss,
            }
            # Mise à jour si la même epoch a déjà une ligne (val arrive parfois après)
            if self.history and self.history[-1]["epoch"] == row["epoch"]:
                prev = self.history[-1]
                for k, v in row.items():
                    if v is not None:
                        prev[k] = v
                row = prev
            else:
                self.history.append(row)
            logger.info(
                "TFT epoch %s | train_quantile_loss=%s | val_quantile_loss=%s",
                row["epoch"],
                f"{row['train_loss']:.6f}" if row["train_loss"] is not None else "n/a",
                f"{row['val_loss']:.6f}" if row["val_loss"] is not None else "n/a",
            )

        def on_validation_epoch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ) -> None:
            metrics = trainer.callback_metrics
            val_loss = _metric_float(
                metrics.get("val_loss") or metrics.get("val_loss_epoch")
            )
            if val_loss is None:
                return
            epoch = int(trainer.current_epoch)
            if self.history and self.history[-1]["epoch"] == epoch:
                self.history[-1]["val_loss"] = val_loss
                self.history[-1]["val_quantile_loss"] = val_loss
            else:
                self.history.append(
                    {
                        "epoch": epoch,
                        "train_loss": None,
                        "val_loss": val_loss,
                        "train_quantile_loss": None,
                        "val_quantile_loss": val_loss,
                    }
                )

    pl.seed_everything(RANDOM_STATE, workers=True)
    encoder_len = int(sequence_length) if sequence_length is not None else int(max_encoder_length)
    enriched, known, resolved_target = prepare_tft_panel(
        df,
        regime_source,
        regime_encoding=regime_encoding,
        target_type=target_type,
        known_reals=known_reals,
    )
    tgt = target_col or resolved_target
    h_cont = _hidden_continuous_size(hidden_size, hidden_continuous_size)

    splits, feature_cols, clip_bounds = prepare_splits(enriched, target_col=tgt)

    train_frame = _prepare_tft_frame(
        splits["train"],
        feature_cols,
        known_reals=known,
        target_col=tgt,
        subsample=max_train_rows,
    )
    if train_frame.empty:
        raise ValueError("Données train TFT vides.")

    training = _build_timeseries_dataset(
        train_frame,
        feature_cols,
        known_reals=known,
        target_col=tgt,
        max_encoder_length=encoder_len,
    )

    train_tickers = set(train_frame["ticker"].unique())
    val_part = splits.get("val", pd.DataFrame())
    val_part = val_part[val_part["ticker"].isin(train_tickers)]
    val_frame = (
        _prepare_tft_frame(val_part, feature_cols, known_reals=known, target_col=tgt)
        if not val_part.empty
        else train_frame.iloc[:0]
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_frame if not val_frame.empty else train_frame,
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # attention_head_size doit diviser hidden_size
    head = int(attention_head_size)
    if hidden_size % head != 0:
        for cand in (1, 2, 4, 8):
            if cand <= hidden_size and hidden_size % cand == 0:
                head = cand
                break

    tft_hparams = {
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "attention_head_size": head,
        "dropout": float(dropout),
        "hidden_continuous_size": int(h_cont),
    }
    logger.info(
        "TFT hparams : %s | epochs≤%s patience=%s | encoder=%s | regime=%s | target=%s",
        tft_hparams,
        max_epochs,
        early_stopping_patience,
        encoder_len,
        regime_encoding,
        tgt,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=tft_hparams["learning_rate"],
        hidden_size=tft_hparams["hidden_size"],
        attention_head_size=tft_hparams["attention_head_size"],
        dropout=tft_hparams["dropout"],
        hidden_continuous_size=tft_hparams["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=TFT_QUANTILES),
        log_interval=-1,
        reduce_on_plateau_patience=2,
    )

    loss_history = _EpochLossHistory()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=early_stopping_patience, mode="min"),
            loss_history,
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(tft, train_loader, val_loader)

    best_val = None
    if loss_history.history:
        vals = [h["val_loss"] for h in loss_history.history if h.get("val_loss") is not None]
        best_val = float(min(vals)) if vals else None

    out_path = model_path or (
        Path(__file__).resolve().parents[1] / "data" / "datasets" / "tft_model.pt"
    )
    if save_model:
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import torch

            torch.save(
                {
                    "state_dict": tft.state_dict(),
                    "hparams": getattr(tft, "hparams", {}),
                    "tft_hparams": tft_hparams,
                    "feature_columns": feature_cols,
                    "known_reals": known,
                    "target_col": tgt,
                    "target_type": target_type,
                    "regime_encoding": regime_encoding,
                    "quantiles": TFT_QUANTILES,
                    "max_encoder_length": encoder_len,
                    "sequence_length": encoder_len,
                    "train_tickers": sorted(train_tickers),
                    "data_version": DATA_VERSION_NOTE,
                    "epoch_history": loss_history.history,
                },
                out_path,
            )
            logger.info("Modèle TFT sauvegardé : %s", out_path)
        except Exception as exc:
            logger.warning("Sauvegarde modèle TFT ignorée : %s", exc)

    return {
        "model": tft,
        "training_dataset": training,
        "feature_columns": feature_cols,
        "known_reals": known,
        "target_col": tgt,
        "target_type": target_type,
        "regime_encoding": regime_encoding,
        "clip_bounds": clip_bounds,
        "max_encoder_length": encoder_len,
        "sequence_length": encoder_len,
        "quantiles": TFT_QUANTILES,
        "train_rows": len(train_frame),
        "train_tickers": sorted(train_tickers),
        "model_path": str(out_path) if save_model else None,
        "epoch_history": loss_history.history,
        "data_version": DATA_VERSION_NOTE,
        "tft_hparams": tft_hparams,
        "best_val_loss": best_val,
    }


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return float(value)
    except Exception:
        return None


def _prediction_time_col(index_df: pd.DataFrame) -> str:
    for cand in ("time_idx_last", "time_idx", "time_idx_first"):
        if cand in index_df.columns:
            return cand
    raise KeyError(f"Colonnes time_idx introuvables dans decoded_index : {index_df.columns.tolist()}")


def _extract_quantile_array(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "output"):
        preds = prediction.output.detach().cpu().numpy()
    elif hasattr(prediction, "detach"):
        preds = prediction.detach().cpu().numpy()
    else:
        preds = np.asarray(prediction)
    if preds.ndim == 3:
        preds = preds[:, 0, :]
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds


def _history_frame_for_split(
    splits: dict[str, pd.DataFrame],
    split_name: str,
    feature_cols: list[str],
    train_tickers: set[str],
    *,
    known_reals: Sequence[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """
    Construit une série continue (train → … → split) pour alimenter l'encodeur TFT,
    puis retourne aussi les dates du split à évaluer.
    """
    idx = _SPLIT_ORDER.index(split_name)
    parts: list[pd.DataFrame] = []
    for name in _SPLIT_ORDER[: idx + 1]:
        part = splits.get(name, pd.DataFrame())
        if part.empty:
            continue
        part = part[part["ticker"].isin(train_tickers)]
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(), set()

    current = splits[split_name]
    current = current[current["ticker"].isin(train_tickers)]
    eval_dates = set(pd.to_datetime(current["date_cours"], errors="coerce").dropna().unique())
    combined = pd.concat(parts, ignore_index=True)
    frame = _prepare_tft_frame(
        combined,
        feature_cols,
        known_reals=known_reals,
        target_col=target_col,
        subsample=None,
    )
    return frame, eval_dates


def predict_tft_quantiles(
    tft_bundle: dict[str, Any],
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    split_name: str,
    known_reals: Sequence[str] | None = None,
    regime_encoding: str | None = None,
    target_type: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Prédit Q10/Q50/Q90 sur toutes les fenêtres valides d'un split temporel."""
    from pytorch_forecasting import TimeSeriesDataSet

    enc = regime_encoding or tft_bundle.get("regime_encoding") or REGIME_ENCODING_ONEHOT
    ttype = target_type or tft_bundle.get("target_type") or TARGET_TYPE_ALPHA
    enriched, known_resolved, resolved_target = prepare_tft_panel(
        df,
        regime_source,
        regime_encoding=enc,
        target_type=ttype,
        known_reals=known_reals if known_reals is not None else tft_bundle.get("known_reals"),
    )
    known = list(known_reals) if known_reals is not None else list(
        tft_bundle.get("known_reals") or known_resolved
    )
    tgt = target_col or tft_bundle.get("target_col") or resolved_target
    splits, feature_cols, _ = prepare_splits(enriched, target_col=tgt)
    train_tickers = set(tft_bundle.get("train_tickers", []))
    frame, eval_dates = _history_frame_for_split(
        splits,
        split_name,
        feature_cols,
        train_tickers,
        known_reals=known,
        target_col=tgt,
    )
    if frame.empty or not eval_dates:
        return pd.DataFrame()

    # Garder alpha_ajuste_risque pour les métriques économiques même si la cible train ≠ alpha
    if TARGET_COLUMN in enriched.columns:
        alpha_map = enriched[["ticker", "date_cours", TARGET_COLUMN]].drop_duplicates(
            ["ticker", "date_cours"], keep="last"
        )
        frame = frame.drop(columns=[TARGET_COLUMN], errors="ignore").merge(
            alpha_map, on=["ticker", "date_cours"], how="left"
        )

    dataset = TimeSeriesDataSet.from_dataset(
        tft_bundle["training_dataset"],
        frame,
        predict=False,
        stop_randomization=True,
    )
    loader = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = tft_bundle["model"].predict(
            loader,
            mode="quantiles",
            return_x=False,
            return_index=True,
        )

    if hasattr(prediction, "output") and hasattr(prediction, "index"):
        preds = _extract_quantile_array(prediction.output)
        index_df = prediction.index.reset_index(drop=True)
    else:
        preds = _extract_quantile_array(prediction)
        index_df = dataset.decoded_index.reset_index(drop=True)
    time_col = _prediction_time_col(index_df)
    n = min(len(index_df), len(preds))
    index_df = index_df.iloc[:n].copy()
    preds = preds[:n]

    merge_cols = ["ticker", "time_idx", "date_cours", tgt]
    if TARGET_COLUMN in frame.columns and TARGET_COLUMN not in merge_cols:
        merge_cols.append(TARGET_COLUMN)
    full_meta = index_df.merge(
        frame[merge_cols],
        left_on=["ticker", time_col],
        right_on=["ticker", "time_idx"],
        how="left",
    )
    full_meta["date_cours"] = pd.to_datetime(full_meta["date_cours"], errors="coerce")
    keep = full_meta["date_cours"].isin(eval_dates).to_numpy()
    full_meta = full_meta.loc[keep].reset_index(drop=True)
    preds = preds[keep]
    if full_meta.empty:
        return pd.DataFrame()

    keep_out = ["ticker", "date_cours", tgt]
    if TARGET_COLUMN in full_meta.columns and TARGET_COLUMN != tgt:
        keep_out.append(TARGET_COLUMN)
    out = full_meta[keep_out].copy()
    n_q = preds.shape[1]
    if n_q >= 3:
        out["q10"], out["q50"], out["q90"] = preds[:, 0], preds[:, 1], preds[:, 2]
    elif n_q == 2:
        out["q10"], out["q50"], out["q90"] = preds[:, 0], preds[:, 1], preds[:, 1]
    else:
        out["q10"] = out["q50"] = out["q90"] = preds[:, 0]
    out["prediction"] = out["q50"]
    out["split"] = split_name
    out["model"] = "tft"
    out = out.drop_duplicates(["ticker", "date_cours"], keep="last")
    spread = (out["q90"] - out["q10"]).abs().mean()
    if spread < 1e-8:
        logger.warning(
            "Quantiles TFT/%s collapsés (q10≈q50≈q90) — vérifier mode=quantiles / QuantileLoss",
            split_name,
        )
    logger.info(
        "Prédictions TFT/%s : %s lignes | %s dates | %s tickers | écart moyen Q90-Q10=%.4f",
        split_name,
        len(out),
        out["date_cours"].nunique(),
        out["ticker"].nunique(),
        spread,
    )
    return out.reset_index(drop=True)


def train_and_evaluate_tft(
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    known_reals: Sequence[str] | None = None,
    regime_encoding: str = REGIME_ENCODING_ONEHOT,
    target_type: str = TARGET_TYPE_ALPHA,
    target_col: str | None = None,
    max_encoder_length: int = DEFAULT_ENCODER_LENGTH,
    sequence_length: int | None = None,
    max_epochs: int = 4,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    learning_rate: float = 0.03,
    hidden_size: int = 16,
    attention_head_size: int = 2,
    dropout: float = 0.1,
    early_stopping_patience: int = 2,
    save_model: bool = True,
) -> dict[str, Any]:
    """Pipeline complet TFT + métriques val/test (IC vs alpha_ajuste_risque)."""
    encoder_len = int(sequence_length) if sequence_length is not None else int(max_encoder_length)
    bundle = train_tft_model(
        df,
        regime_source,
        known_reals=known_reals,
        regime_encoding=regime_encoding,
        target_type=target_type,
        target_col=target_col,
        max_encoder_length=encoder_len,
        max_epochs=max_epochs,
        max_train_rows=max_train_rows,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        early_stopping_patience=early_stopping_patience,
        save_model=save_model,
    )
    known = list(bundle.get("known_reals") or REGIME_COLUMNS)
    train_tgt = bundle.get("target_col") or TARGET_COLUMN
    # Métriques reportées toujours vs alpha pour comparabilité inter-modèles
    eval_tgt = TARGET_COLUMN

    predictions: list[pd.DataFrame] = []
    results: dict[str, Any] = {
        "model": "tft",
        "target_col": train_tgt,
        "target_type": bundle.get("target_type", target_type),
        "eval_target_col": eval_tgt,
        "regime_encoding": bundle.get("regime_encoding", regime_encoding),
        "sequence_length": bundle.get("sequence_length", encoder_len),
        "quantiles": TFT_QUANTILES,
        "known_reals": known,
        "train_rows": bundle["train_rows"],
        "clip_bounds": bundle["clip_bounds"],
        "epoch_history": bundle.get("epoch_history", []),
        "data_version": bundle.get("data_version", DATA_VERSION_NOTE),
        "tft_hparams": bundle.get("tft_hparams"),
        "best_val_loss": bundle.get("best_val_loss"),
        "splits": {},
    }

    for split_name in ("val", "test"):
        pred_df = predict_tft_quantiles(
            bundle,
            df,
            regime_source,
            split_name=split_name,
            known_reals=known,
            regime_encoding=bundle.get("regime_encoding", regime_encoding),
            target_type=bundle.get("target_type", target_type),
            target_col=train_tgt,
        )
        if pred_df.empty:
            continue
        if eval_tgt not in pred_df.columns:
            logger.warning(
                "Colonne d'évaluation %s absente — fallback sur %s",
                eval_tgt,
                train_tgt,
            )
            metrics = evaluate_scoring_model(pred_df, "prediction", train_tgt)
        else:
            metrics = evaluate_scoring_model(pred_df, "prediction", eval_tgt)
        # Garde-fou fuite / aberrant
        ic = metrics.get("ic_mean")
        if ic is not None and abs(float(ic)) > 0.5:
            logger.error(
                "IC %s=%.4f aberrant (|IC|>0.5) — probable fuite de données. STOP.",
                split_name,
                ic,
            )
            results["anomaly"] = {
                "split": split_name,
                "ic_mean": float(ic),
                "message": "|IC| > 0.5 — vérifier fuite de données",
            }
        results["splits"][split_name] = {"rows": len(pred_df), "metrics": metrics}
        predictions.append(pred_df)
        logger.info(
            "[tft/%s] IC=%.4f | Hit=%.1f%% | Sharpe=%.2f | train_target=%s eval=%s",
            split_name,
            metrics.get("ic_mean", np.nan),
            (metrics.get("hit_ratio", np.nan) or 0) * 100,
            metrics.get("sharpe_long_short", np.nan),
            train_tgt,
            eval_tgt,
        )

    if predictions:
        results["predictions"] = pd.concat(predictions, ignore_index=True)
    return results


def optimize_hyperparameters(
    df: pd.DataFrame,
    regime_source: pd.DataFrame,
    *,
    known_reals: Sequence[str] | None = None,
    regime_encoding: str = REGIME_ENCODING_ONEHOT,
    target_type: str = TARGET_TYPE_ALPHA,
    max_encoder_length: int = DEFAULT_ENCODER_LENGTH,
    sequence_length: int | None = None,
    n_trials: int = 25,
    max_epochs: int = 15,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    early_stopping_patience: int = 5,
    study_name: str = "tft_threshold_regime",
    hidden_sizes: Sequence[int] | None = None,
    dropouts: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Recherche Optuna sur validation 2021-2022 (val_loss).

    Espaces (défaut élargi vers le bas pour petits panels) :
      hidden_size ∈ {2,4,8,16,32,64}  (128 exclu — trop lent hors GPU)
      attention_head_size ∈ {1,2,4}
      dropout ∈ {0.1,0.2,0.3,0.4,0.5}
      learning_rate ∈ [1e-4, 1e-2] (log)
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError(
            "optuna requis : pip install optuna"
        ) from exc

    encoder_len = int(sequence_length) if sequence_length is not None else int(max_encoder_length)
    hs_choices = list(hidden_sizes) if hidden_sizes is not None else list(DEFAULT_HIDDEN_SIZES)
    drop_choices = list(dropouts) if dropouts is not None else list(DEFAULT_DROPOUTS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        hidden_size = trial.suggest_categorical("hidden_size", hs_choices)
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
        dropout = trial.suggest_categorical("dropout", drop_choices)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Contrainte : head doit diviser hidden_size
        if hidden_size % int(attention_head_size) != 0:
            raise optuna.TrialPruned()

        logger.info(
            "Optuna trial %s | hs=%s heads=%s drop=%s lr=%.5f | enc=%s regime=%s target=%s",
            trial.number,
            hidden_size,
            attention_head_size,
            dropout,
            learning_rate,
            encoder_len,
            regime_encoding,
            target_type,
        )
        bundle = train_tft_model(
            df,
            regime_source,
            known_reals=known_reals,
            regime_encoding=regime_encoding,
            target_type=target_type,
            max_encoder_length=encoder_len,
            max_epochs=max_epochs,
            max_train_rows=max_train_rows,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            early_stopping_patience=early_stopping_patience,
            save_model=False,
        )
        best_val = bundle.get("best_val_loss")
        if best_val is None or not np.isfinite(best_val):
            raise optuna.TrialPruned()
        trial.set_user_attr("epoch_history", bundle.get("epoch_history", []))
        return float(best_val)

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(
        "Optuna best trial #%s val_loss=%.6f params=%s",
        best.number,
        best.value,
        best.params,
    )
    return {
        "best_params": dict(best.params),
        "best_val_loss": float(best.value) if best.value is not None else None,
        "best_trial_number": int(best.number),
        "n_trials": n_trials,
        "max_epochs": max_epochs,
        "max_train_rows": max_train_rows,
        "hidden_sizes": hs_choices,
        "dropouts": drop_choices,
        "early_stopping_patience": early_stopping_patience,
        "regime_encoding": regime_encoding,
        "target_type": target_type,
        "sequence_length": encoder_len,
        "known_reals": list(known_reals) if known_reals is not None else None,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": dict(t.params),
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }


def compare_with_baselines(
    tft_results: dict[str, Any],
    baseline_report_path: Path,
) -> dict[str, Any]:
    """Compare TFT (test) aux baselines de l'étape 5."""
    import json

    if not baseline_report_path.is_file():
        return {"error": "step5_validation_report.json introuvable"}

    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    leaderboard = baseline.get("leaderboard", [])
    if isinstance(leaderboard, list):
        baseline_test = {row["model"]: row for row in leaderboard if row.get("split") == "test"}
    else:
        baseline_test = {}

    tft_test = tft_results.get("splits", {}).get("test", {}).get("metrics", {})
    comparison = {
        "tft": {
            "ic_mean": tft_test.get("ic_mean"),
            "hit_ratio": tft_test.get("hit_ratio"),
            "sharpe_long_short": tft_test.get("sharpe_long_short"),
        },
        "baselines": baseline_test,
    }

    best_base_ic = max(
        (v.get("ic_mean") or -999 for v in baseline_test.values()),
        default=-999,
    )
    tft_ic = tft_test.get("ic_mean") or -999
    comparison["winner_ic"] = "tft" if tft_ic >= best_base_ic else max(
        baseline_test,
        key=lambda k: baseline_test[k].get("ic_mean") or -999,
        default="tft",
    )
    return comparison
