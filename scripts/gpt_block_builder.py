# scripts/gpt_block_builder.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_llm_records(path: str) -> Dict[str, dict]:
    recs: Dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = str(rec.get("Ticker"))
            recs[t] = rec
    return recs


def _load_stage2_vals(path: str) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df["ticker"] = df["ticker"].astype(str)
    vals: Dict[str, dict] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        vals[t] = {
            "PE": row.get("val_PE", np.nan),
            "PEG": row.get("val_PEG", np.nan),
            "YoY": row.get("val_YoY", np.nan),
        }
    return vals


def _safe_float(v, default="N/A", fmt=None):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if fmt is None:
        return x
    return fmt.format(x)


def _trend_tag(price, ema50, ema200):
    try:
        price = float(price)
        ema50 = float(ema50)
        ema200 = float(ema200)
    except Exception:
        return "Unknown"
    if price > ema50 > ema200:
        return "StrongUp"
    if price > ema200 and ema50 > ema200:
        return "Up"
    if price < ema50 < ema200:
        return "Down"
    return "Sideways"


def build_gpt_blocks(
    top_tickers: List[str],
    brain_scores: Dict[str, float],
    llm_data_path: str = "data/LLM_today_data.jsonl",
    stage2_path: str = "data/stage2_merged.csv",
    news_blocks: Dict[str, str] | None = None,
) -> str:
    """
    Build compact text blocks for GPT-5, one block per ticker.
    """
    llm_recs = _load_llm_records(llm_data_path)
    vals_map = _load_stage2_vals(stage2_path)
    news_blocks = news_blocks or {}

    blocks: List[str] = []

    for rank, t in enumerate(top_tickers, start=1):
        rec = llm_recs.get(t)
        if rec is None:
            # skip silently if missing from LLM data
            continue

        brain_score = brain_scores.get(t, 0.0)

        price = rec.get("current_price")
        vol30 = rec.get("volatility_30")
        rsi14 = rec.get("RSI14")
        macd = rec.get("MACD")
        momentum = rec.get("Momentum")
        atrp = rec.get("ATR%")
        avg_low_30 = rec.get("avg_low_30")
        avg_high_30 = rec.get("avg_high_30")
        pos_30d = rec.get("pos_30d")
        ema50 = rec.get("EMA50")
        ema200 = rec.get("EMA200")
        mkt_trend = rec.get("MarketTrend", "Unknown")

        trend_tag = _trend_tag(price, ema50, ema200)

        vals = vals_map.get(t, {})
        pe = vals.get("PE", None)
        peg = vals.get("PEG", None)

        news_text = news_blocks.get(t, "N/A")

        block = []
        block.append(f"### TICKER: {t}")
        block.append(f"BrainScore: {brain_score:.2f} (rank {rank}/{len(top_tickers)})")
        block.append("Core metrics:")
        block.append(
            f"- Price: {_safe_float(price, 'N/A', '{:.2f}')}"
        )
        block.append(
            f"- 30D Volatility: {_safe_float(vol30, 'N/A', '{:.4f}')}"
        )
        block.append(f"- RSI14: {_safe_float(rsi14, 'N/A', '{:.2f}')}")
        block.append(f"- MACD: {_safe_float(macd, 'N/A', '{:.3f}')}")
        block.append(f"- Momentum_10D: {_safe_float(momentum, 'N/A', '{:.2f}')}")
        block.append(f"- ATR%: {_safe_float(atrp, 'N/A', '{:.2f}')}")
        block.append(
            f"- 30D Range Position: {_safe_float(pos_30d, 'N/A', '{:.2f}')}"
        )
        block.append(f"- TrendTag: {trend_tag}")
        block.append(f"- MarketTrend: {mkt_trend}")
        block.append(
            f"- Valuation_PE: {_safe_float(pe, 'N/A', '{:.2f}')}, PEG: {_safe_float(peg, 'N/A', '{:.2f}')}"
        )

        if news_text == "N/A":
            block.append("Brief News:\n- N/A")
        else:
            block.append(f"Brief News:\n{news_text}")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)
