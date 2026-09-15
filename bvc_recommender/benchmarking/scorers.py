"""
Scorers Niveau 2 — remplacent uniquement la Composante B (TFT).

Entraînement : train ≤ 2020, validation ≤ 2022 (même splits que RAF-ADAPT).
Features : colonnes ``*_z`` du ml_dataset (aplaties pour Ridge/LightGBM).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bvc_recommender.benchmarking.config import RANDOM_SEED
from bvc_recommender.config import DATASET_DIR, RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.baseline_models import prepare_splits, train_model

logger = logging.getLogger(__name__)

ScorerName = Literal["ridge", "lightgbm", "lstm"]


def load_ml_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "ml_dataset.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"ml_dataset manquant : {path}")
    return pd.read_parquet(path)


def train_flat_scorer(name: Literal["ridge", "lightgbm"]) -> tuple[Any, list[str], pd.DataFrame]:
    """Entraîne Ridge ou LightGBM ; retourne (modèle, features, dataset annoté)."""
    df = load_ml_dataset()
    splits, features, _ = prepare_splits(df)
    train = splits["train"]
    if train.empty:
        raise ValueError("Split train vide.")
    X_train = train[features].fillna(0.0)
    y_train = train[TARGET_COLUMN]
    model = train_model(name, X_train, y_train)
    logger.info("[%s] entraîné | features=%s | n_train=%s", name, len(features), len(train))
    return model, features, df


def predict_flat_panel(model: Any, features: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Panel ticker × date avec prediction (val+test)."""
    data = df[df["split"].isin(["val", "test"])].copy()
    X = data[features].fillna(0.0)
    data["prediction"] = model.predict(X)
    return data[["ticker", "date_cours", "prediction", "split"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


class _LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _build_sequences(
    df: pd.DataFrame,
    features: list[str],
    *,
    lookback: int,
    split_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construit des fenêtres (lookback, n_features) par ticker."""
    data = df.copy()
    data["date_cours"] = pd.to_datetime(data["date_cours"], errors="coerce")
    data = data.dropna(subset=["date_cours", TARGET_COLUMN]).sort_values(["ticker", "date_cours"])

    xs, ys, meta_rows = [], [], []
    for ticker, g in data.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        vals = g[features].fillna(0.0).to_numpy(dtype=np.float32)
        targets = g[TARGET_COLUMN].to_numpy(dtype=np.float32)
        splits = g["split"].tolist()
        dates = g["date_cours"].tolist()
        for i in range(lookback - 1, len(g)):
            if split_filter is not None and splits[i] not in split_filter:
                continue
            xs.append(vals[i - lookback + 1 : i + 1])
            ys.append(targets[i])
            meta_rows.append({"ticker": ticker, "date_cours": dates[i], "split": splits[i]})

    if not xs:
        return (
            np.empty((0, lookback, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(columns=["ticker", "date_cours", "split"]),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta_rows)


def train_lstm_scorer(
    *,
    lookback: int = 20,
    hidden: int = 32,
    epochs: int = 8,
    batch_size: int = 256,
    max_train_seq: int = 40_000,
) -> tuple[_LSTMRegressor, list[str], pd.DataFrame, int]:
    """
    LSTM 1 couche + dense — pas d'attention, pas de régime.
    Prédiction ponctuelle (pas de quantiles).
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = load_ml_dataset()
    _, features, _ = prepare_splits(df)
    X_tr, y_tr, _ = _build_sequences(df, features, lookback=lookback, split_filter={"train"})
    if len(X_tr) == 0:
        raise ValueError("Aucune séquence LSTM en train.")

    if len(X_tr) > max_train_seq:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_tr), size=max_train_seq, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    device = torch.device("cpu")
    model = _LSTMRegressor(n_features=len(features), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        logger.info("[lstm] epoch %s/%s | loss=%.5f", ep + 1, epochs, total / max(n, 1))

    return model, features, df, lookback


@torch.no_grad()
def predict_lstm_panel(
    model: _LSTMRegressor,
    features: list[str],
    df: pd.DataFrame,
    *,
    lookback: int,
    batch_size: int = 512,
) -> pd.DataFrame:
    model.eval()
    X, _, meta = _build_sequences(
        df, features, lookback=lookback, split_filter={"val", "test"}
    )
    if len(X) == 0:
        return pd.DataFrame(columns=["ticker", "date_cours", "prediction", "split"])

    device = next(model.parameters()).device
    preds: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        preds.append(model(xb).cpu().numpy())
    meta = meta.copy()
    meta["prediction"] = np.concatenate(preds)
    return meta[["ticker", "date_cours", "prediction", "split"]]


def build_scorer_predictions(name: ScorerName) -> pd.DataFrame:
    """Point d'entrée : produit le panel de scores pour un modèle Niveau 2."""
    if name in ("ridge", "lightgbm"):
        model, features, df = train_flat_scorer(name)  # type: ignore[arg-type]
        return predict_flat_panel(model, features, df)

    if name == "lstm":
        model, features, df, lookback = train_lstm_scorer()
        return predict_lstm_panel(model, features, df, lookback=lookback)

    raise ValueError(f"Scorer inconnu : {name}")
