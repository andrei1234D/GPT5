# scripts/scorebot_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_1d(preds: Any, n: int | None = None) -> np.ndarray:
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
        # multiclass_top1 -> reduce to one score per row
        out = np.max(arr.astype(float, copy=False), axis=1)
        if n is not None and out.shape[0] != n:
            raise ValueError(f"Prediction length mismatch: got {out.shape[0]}, expected {n}")
        return out

    raise ValueError(f"Unsupported prediction shape: {arr.shape}")


def _resolve_model_ref(
    m: Any,
    model_root: Path | None,
    cache: dict,
    registry: Mapping[str, Any] | None,
) -> Any:
    """
    Resolve model reference:
      - if estimator object -> return as-is
      - if string:
          1) try registry lookup (alias -> estimator)
          2) else treat as path and joblib.load() it
    """
    if not isinstance(m, str):
        return m

    # 1) Registry alias
    if registry is not None and m in registry:
        return registry[m]

    # 2) Cached path load
    if m in cache:
        return cache[m]

    import joblib  # lazy import

    p = Path(m)
    if not p.is_absolute() and model_root is not None:
        p = (model_root / p).resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Model reference '{m}' not found as alias in registry and not found as file path "
            f"(resolved: {p})."
        )

    loaded = joblib.load(str(p))
    cache[m] = loaded
    return loaded


@dataclass
class ScoreBotSlim:
    models: Sequence[Any]
    weights: Sequence[float]
    feature_cols: Sequence[str]

    # Optional inference-time helpers (may be missing on old pickles)
    model_root: Path | None = None
    model_registry: Mapping[str, Any] | None = None

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        # Backward-compatible for old pickles
        if not hasattr(self, "_model_cache") or getattr(self, "_model_cache") is None:
            setattr(self, "_model_cache", {})
        if not hasattr(self, "model_root"):
            setattr(self, "model_root", None)
        if not hasattr(self, "model_registry"):
            setattr(self, "model_registry", None)

        cache = getattr(self, "_model_cache")
        root = getattr(self, "model_root")
        registry = getattr(self, "model_registry")

        X = df.loc[:, list(self.feature_cols)]
        n = len(X)

        final: np.ndarray | None = None

        for model, w in zip(self.models, self.weights):
            model_obj = _resolve_model_ref(model, root, cache, registry)

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
