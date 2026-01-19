from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _ensure_1d(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == n:
        return arr
    raise ValueError(f"Expected 1D array of length {n}, got {arr.shape}")


def _proba_to_score(
    proba: np.ndarray,
    reps: np.ndarray,
) -> np.ndarray:
    """
    EXACT training logic:
    score = sum_c p(c) * rep[c]
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(reps):
        raise ValueError(f"Bad proba shape {proba.shape}, reps={reps}")
    return (proba * reps.reshape(1, -1)).sum(axis=1)


def _build_reps(thr: dict) -> np.ndarray:
    mid = thr["mid_thr"]
    high = thr["high_thr"]
    return np.array(
        [mid - 50.0, (mid + high) / 2.0, high + 50.0],
        dtype=float,
    )


# -----------------------------
# ScoreBotSlim (RAW score only)
# -----------------------------
@dataclass
class ScoreBotSlim:
    models: Dict[str, Any]
    feature_cols: list[str]
    weights: Dict[str, float]
    reps_by_model: Dict[str, list] | None = None
    thr_by_model: Dict[str, dict] | None = None

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        X = df.loc[:, self.feature_cols]
        n = len(X)
        out = np.zeros(n, dtype=float)

        for name, model in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue

            # ---- MULTICLASS (training path) ----
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)

                if self.reps_by_model and name in self.reps_by_model:
                    reps = np.asarray(self.reps_by_model[name], dtype=float)
                elif self.thr_by_model and name in self.thr_by_model:
                    reps = _build_reps(self.thr_by_model[name])
                else:
                    raise RuntimeError(f"Missing reps for model '{name}'")

                s = _proba_to_score(proba, reps)

            # ---- REGRESSION (fallback) ----
            else:
                preds = model.predict(X)
                s = _ensure_1d(preds, n)

            out += w * s

        return pd.Series(out, index=df.index, name="raw_score")

    def predict_one(self, feature_dict: dict) -> float:
        row = pd.DataFrame([feature_dict]).loc[:, self.feature_cols]
        return float(self.predict_df(row).iloc[0])


# -----------------------------
# Calibrated wrapper
# -----------------------------
@dataclass
class CalibratedScoreBot:
    base_bot: Any
    calibrator_name: str | None = None
    calibrator_params: dict | None = None
    calibrator: Any = None
    clip_low: float = 0.0
    clip_high: float = 1000.0

    @property
    def feature_cols(self) -> list[str]:
        return list(getattr(self.base_bot, "feature_cols", []))

    def _apply_cal(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=float).reshape(-1)
        if not hasattr(self.calibrator, "predict"):
            raise RuntimeError("Invalid calibrator")
        y = self.calibrator.predict(x)
        return np.clip(y, self.clip_low, self.clip_high)

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base_bot.predict_df(df).to_numpy()
        cal = self._apply_cal(raw)
        return pd.Series(cal, index=df.index, name="pred_score")

    def predict_one(self, feature_dict: dict) -> float:
        return float(self.predict_df(pd.DataFrame([feature_dict])).iloc[0])


# Backward alias
CalibratedScoreBotSlim = CalibratedScoreBot
