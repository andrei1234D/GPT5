# scripts/scorebot_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _as_1d(preds: Any, n: int | None = None) -> np.ndarray:
    """Normalize predictions to a 1D float array of length n.

    Supports:
      - (n,) regression outputs
      - (n, k) multiclass probabilities/logits -> reduces to TOP-1 via max(axis=1)
      - pandas Series/DataFrame
    """
    if isinstance(preds, pd.DataFrame):
        arr = preds.values
    elif isinstance(preds, pd.Series):
        arr = preds.values
    else:
        arr = np.asarray(preds)

    if arr.ndim == 0:
        if n is None:
            raise ValueError("Scalar prediction with unknown n_samples.")
        return np.full((n,), float(arr), dtype=float)

    if arr.ndim == 1:
        out = arr.astype(float, copy=False)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Prediction length mismatch: got {out.shape[0]}, expected {n}")
        return out

    if arr.ndim == 2:
        # Artifact is named 'multiclass_top1' -> use TOP-1 reduction.
        out = np.max(arr.astype(float, copy=False), axis=1)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Prediction length mismatch: got {out.shape[0]}, expected {n}")
        return out

    raise ValueError(f"Unsupported prediction shape: {arr.shape}")


@dataclass
class ScoreBotSlim:
    models: Sequence[Any]
    weights: Sequence[float]
    feature_cols: Sequence[str]

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        X = df.loc[:, list(self.feature_cols)]
        n = len(X)

        final: np.ndarray | None = None

        for model, w in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)

            p1 = _as_1d(preds, n=n)
            if final is None:
                final = np.zeros((n,), dtype=float)
            final += float(w) * p1

        if final is None:
            final = np.zeros((n,), dtype=float)

        return pd.Series(final, index=df.index, name="raw_score")


@dataclass
class CalibratedScoreBotSlim:
    base_bot: Any
    calibrator: Any
    clip_low: float = 0.0
    clip_high: float = 1000.0

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base_bot.predict_df(df).values
        raw_1d = _as_1d(raw, n=len(df))

        if hasattr(self.calibrator, "predict_proba"):
            cal = self.calibrator.predict_proba(raw_1d.reshape(-1, 1))
            cal_1d = _as_1d(cal, n=len(df))
        elif hasattr(self.calibrator, "predict"):
            cal = self.calibrator.predict(raw_1d.reshape(-1, 1))
            cal_1d = _as_1d(cal, n=len(df))
        else:
            cal_1d = _as_1d(self.calibrator(raw_1d), n=len(df))

        cal_1d = np.clip(cal_1d.astype(float, copy=False), self.clip_low, self.clip_high)
        return pd.Series(cal_1d, index=df.index, name="pred_score")
