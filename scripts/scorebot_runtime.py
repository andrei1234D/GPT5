# scripts/scorebot_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _as_1d(preds: Any, n: int | None = None) -> np.ndarray:
    """Normalize predictions to a 1D float array of length n."""
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
        # multiclass_top1 -> reduce to 1 score per row
        out = np.max(arr.astype(float, copy=False), axis=1)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Prediction length mismatch: got {out.shape[0]}, expected {n}")
        return out

    raise ValueError(f"Unsupported prediction shape: {arr.shape}")


def _maybe_load_model(m: Any, model_root: Path | None, cache: dict) -> Any:
    """If model is a string path, load it once (joblib) relative to model_root."""
    if not isinstance(m, str):
        return m

    if m in cache:
        return cache[m]

    import joblib  # lazy import

    p = Path(m)
    if not p.is_absolute() and model_root is not None:
        p = (model_root / p).resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Model reference is a string but file was not found: '{m}' (resolved: {p})"
        )

    loaded = joblib.load(str(p))
    cache[m] = loaded
    return loaded


@dataclass
class ScoreBotSlim:
    models: Sequence[Any]
    weights: Sequence[float]
    feature_cols: Sequence[str]
    # NOTE: do NOT rely on this existing on old pickles; brain_ranker will set it if available
    model_root: Path | None = None

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        # Backward-compatible: old pickles won't have these attrs
        if not hasattr(self, "_model_cache") or getattr(self, "_model_cache") is None:
            setattr(self, "_model_cache", {})
        if not hasattr(self, "model_root"):
            setattr(self, "model_root", None)

        cache = getattr(self, "_model_cache")
        X = df.loc[:, list(self.feature_cols)]
        n = len(X)

        final: np.ndarray | None = None

        for model, w in zip(self.models, self.weights):
            model_obj = _maybe_load_model(model, getattr(self, "model_root", None), cache)

            if hasattr(model_obj, "predict_proba"):
                preds = model_obj.predict_proba(X)
            else:
                preds = model_obj.predict(X)

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
