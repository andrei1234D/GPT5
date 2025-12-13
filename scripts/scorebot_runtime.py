# scripts/scorebot_runtime.py
from __future__ import annotations

import numpy as np
import pandas as pd


class ScoreBotSlim:
    """
    Runtime-only ensemble.
    Stores ONLY models + weights + feature_cols.
    """
    def __init__(self, models: dict, feature_cols, weights: dict):
        self.models = dict(models)
        self.feature_cols = list(feature_cols)
        self.weights = dict(weights)

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        final = np.zeros(len(df), dtype=float)
        for name, model in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue
            preds = model.predict(X)
            final += w * np.asarray(preds, dtype=float)
        return pd.Series(final, index=df.index)

    def predict_one(self, feature_dict: dict) -> float:
        row = pd.DataFrame([feature_dict])[self.feature_cols]
        final = 0.0
        for name, model in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue
            final += w * float(model.predict(row)[0])
        return float(final)


class CalibratedScoreBotSlim:
    """
    Wraps a ScoreBotSlim and applies a fitted calibration function to raw predictions.
    """
    def __init__(self, base_bot, calibrator_name, calibrator_params, calibrator_obj_or_fn):
        self.base_bot = base_bot
        self.calibrator_name = calibrator_name
        self.calibrator_params = calibrator_params
        self.calibrator = calibrator_obj_or_fn
        self.feature_cols = getattr(base_bot, "feature_cols", None)

    def _apply_cal(self, raw_preds: np.ndarray) -> np.ndarray:
        raw_preds = np.asarray(raw_preds, dtype=float).reshape(-1)
        if hasattr(self.calibrator, "predict"):
            out = self.calibrator.predict(raw_preds)
        else:
            out = self.calibrator(raw_preds)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1000.0)

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base_bot.predict_df(df).values
        cal = self._apply_cal(raw)
        return pd.Series(cal, index=df.index)

    def predict_one(self, feature_dict: dict) -> float:
        raw = float(self.base_bot.predict_one(feature_dict))
        cal = float(self._apply_cal(np.array([raw]))[0])
        return cal
