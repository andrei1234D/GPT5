# scripts/scorebot_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


def _as_1d(x: Any, n: int | None = None) -> np.ndarray:
    """Coerce model outputs to 1D float array length n."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 0:
        if n is None:
            raise ValueError("Scalar output with unknown n")
        return np.full((n,), float(arr), dtype=float)

    if arr.ndim == 1:
        out = arr.astype(float, copy=False)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Length mismatch: got {out.shape[0]}, expected {n}")
        return out

    if arr.ndim == 2:
        # Prefer binary proba P(class=1) when available.
        if arr.shape[1] >= 2:
            out = arr[:, 1].astype(float, copy=False)
        else:
            out = arr.reshape(-1).astype(float, copy=False)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Length mismatch: got {out.shape[0]}, expected {n}")
        return out

    raise ValueError(f"Unsupported output shape: {arr.shape}")


def _model_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return a continuous per-row score from an estimator.

    - If predict_proba exists, returns P(class=1) (preferred for classifiers).
    - Else falls back to predict().
    """
    n = len(X)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return _as_1d(proba, n=n)
        except Exception:
            pass
    preds = model.predict(X)
    return _as_1d(preds, n=n)


@dataclass
class ScoreBotSlim:
    """Matches your SLIM pickle: models/weights are dicts."""
    models: Dict[str, Any]
    feature_cols: list[str]
    weights: Dict[str, float]

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        X = df.loc[:, self.feature_cols]
        out = np.zeros(len(X), dtype=float)

        for name, model in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue
            out += w * _model_score(model, X)

        return pd.Series(out, index=df.index, name="raw_score")

    def predict_one(self, feature_dict: dict) -> float:
        row = pd.DataFrame([feature_dict]).loc[:, self.feature_cols]
        s = 0.0
        for name, model in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue
            s += w * float(_model_score(model, row)[0])
        return float(s)


@dataclass
class CalibratedScoreBot:
    """Matches your SLIM pickle (IsotonicRegression calibrator)."""
    base_bot: Any
    calibrator_name: str | None = None
    calibrator_params: dict | None = None
    calibrator: Any = None
    clip_low: float = 0.0
    clip_high: float = 1000.0

    @property
    def feature_cols(self) -> list[str]:
        return list(getattr(self.base_bot, "feature_cols", []))

    def _apply_cal(self, raw_1d: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_1d, dtype=float).reshape(-1)
        if self.calibrator is None:
            raise RuntimeError("Missing calibrator on CalibratedScoreBot")

        if hasattr(self.calibrator, "predict"):
            y = self.calibrator.predict(x)
        else:
            y = self.calibrator(x)

        y = np.asarray(y, dtype=float).reshape(-1)
        return np.clip(y, self.clip_low, self.clip_high)

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base_bot.predict_df(df).to_numpy()
        cal = self._apply_cal(raw)
        return pd.Series(cal, index=df.index, name="pred_score")

    def predict_one(self, feature_dict: dict) -> float:
        raw = float(self.base_bot.predict_one(feature_dict))
        return float(self._apply_cal(np.asarray([raw]))[0])


# Backwards-compatible aliases for older imports
CalibratedScoreBotSlim = CalibratedScoreBot
