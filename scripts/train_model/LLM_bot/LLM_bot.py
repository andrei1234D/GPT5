# -*- coding: utf-8 -*-
"""
train_llm_signal_regression.py — Resume-Safe LLM Buy Strength Predictor
------------------------------------------------------------------------
Trains DistilBERT to predict a continuous buy score (0–1000).
Automatically resumes from last checkpoint.
Logs MSE and MAE per epoch and saves training progress.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH = Path("../LLM_Training_data/LLM_Training_data_with_response.parquet")
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("Brain/llm_signal_regression_model")
HISTORY_DIR = Path("Training_history")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
print(f"[INFO] Loading dataset from {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"[INFO] Loaded {len(df):,} rows.")

# Use only rows with valid numeric scores
df = df[df["score"].notna()]
print(f"[INFO] {len(df):,} samples with valid scores.")

# ==========================================================
# BUILD PROMPTS
# ==========================================================
def build_prompt(row):
    return (
        f"TICKER: {row['Ticker']}\n"
        f"DATE: {row['Date']}\n"
        f"PRICE: {row['Close']:.2f}\n"
        f"RSI14: {row.get('RSI14', np.nan):.2f}\n"
        f"MACD: {row.get('MACD', np.nan):.3f}\n"
        f"SMA50: {row.get('SMA50', np.nan):.2f}\n"
        f"Volatility: {row.get('Volatility', np.nan):.4f}\n"
        f"Momentum: {row.get('Momentum', np.nan):.2f}\n"
        f"MarketTrend: {row.get('MarketTrend', 'Unknown')}"
    )

df["prompt"] = df.apply(build_prompt, axis=1)

# ==========================================================
# TOKENIZATION
# ==========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = Dataset.from_pandas(df[["prompt", "score"]])

def tokenize_fn(batch):
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.rename_column("score", "labels")
tokenized = tokenized.train_test_split(test_size=0.2, seed=42)

# ==========================================================
# MODEL (REGRESSION)
# ==========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression"
)

# ==========================================================
# METRICS
# ==========================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {"MSE": mse, "MAE": mae}

# ==========================================================
# TRAINING HISTORY LOGGER
# ==========================================================
EPOCH_LOG_PATH = HISTORY_DIR / "epoch_metrics_regression.csv"

class EpochLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "MSE", "MAE", "loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            mse = round(metrics.get("eval_MSE", 0), 4)
            mae = round(metrics.get("eval_MAE", 0), 4)
            loss = round(metrics.get("eval_loss", 0), 4)
            print(f"[EPOCH {int(state.epoch)}] MSE: {mse:.4f} | MAE: {mae:.4f} | Loss: {loss}")
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([state.epoch, mse, mae, loss])

# ==========================================================
# TRAINING
# ==========================================================
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=str(OUTPUT_DIR / "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="MSE",
    save_total_limit=3,
    resume_from_checkpoint=True
)

# Detect latest checkpoint
latest_ckpt = None
ckpts = sorted([p for p in OUTPUT_DIR.glob("checkpoint-*") if p.is_dir()],
               key=lambda x: int(x.name.split("-")[-1]))
if ckpts:
    latest_ckpt = str(ckpts[-1])
    print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
else:
    print("[INFO] No existing checkpoint found, starting fresh.")

epoch_logger = EpochLoggerCallback(EPOCH_LOG_PATH)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[epoch_logger],
)

trainer.train(resume_from_checkpoint=latest_ckpt)

# ==========================================================
# EVALUATION
# ==========================================================
eval_result = trainer.evaluate()
print(f"[RESULTS] {eval_result}")

# ==========================================================
# SAVE MODEL ("Brain")
# ==========================================================
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"[SAVED] Regression model saved to {OUTPUT_DIR}")

# ==========================================================
# PLOT TRAINING HISTORY
# ==========================================================
log_df = pd.read_csv(EPOCH_LOG_PATH)
plt.figure(figsize=(8, 5))
plt.plot(log_df["epoch"], log_df["MSE"], marker="o", label="MSE")
plt.plot(log_df["epoch"], log_df["MAE"], marker="s", label="MAE")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Progress (Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(HISTORY_DIR / "training_progress_regression.png")
print(f"[SAVED] Training metrics plot -> {HISTORY_DIR / 'training_progress_regression.png'}")
