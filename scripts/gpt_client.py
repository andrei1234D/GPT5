# scripts/gpt_client.py
import os
import re
import time
import json
import logging
from typing import Optional, Dict, Any, List

import requests

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("gpt_client")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("GPT_CLIENT_LOG_LEVEL", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# -------- Robust extractor for the Responses API --------
def extract_output_text(data: Dict[str, Any]) -> Optional[str]:
    """
    Tries several shapes the Responses API may return:
      - data["output_text"] (string)
      - data["output"][i]["content"][j]["text"] or ["value"] when type in {"output_text","input_text","text"}
    Returns a single concatenated string or None.
    """
    if not isinstance(data, dict):
        return None

    # Easiest path first
    txt = data.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = data.get("output")
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                # Some SDKs return type="output_text" at top-level of 'output'
                maybe_txt = item.get("text") or item.get("value")
                if isinstance(maybe_txt, str):
                    chunks.append(maybe_txt)
                continue
            for seg in item.get("content", []) or []:
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") in {"output_text", "input_text", "text"}:
                    t = seg.get("text") or seg.get("value")
                    if isinstance(t, str):
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()

    # Last resort: stringify
    return None


# -------- OpenAI call wrapper (Responses API) --------
def call_gpt5(
    system_msg: str,
    user_msg: str,
    *,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    timeout: Optional[float] = None,
    retries: int = 2,
) -> str:
    """
    Calls the OpenAI Responses API with a per-call timeout and simple retry.
    Raises on failure (no silent fallbacks).

    Env defaults:
      OPENAI_MODEL       (default "gpt-5")
      OPENAI_TEMPERATURE (default "0.2")
      OPENAI_MAX_TOKENS  (default "1200")
      OPENAI_TIMEOUT     (seconds; default "180")
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_MODEL", "gpt-5")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2")) if temperature is None else float(temperature)
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1200")) if max_tokens is None else int(max_tokens)
    timeout = float(os.getenv("OPENAI_TIMEOUT", "180")) if timeout is None else float(timeout)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Prefer messages-style input for clearer separation of system/user
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }

    url = f"{OPENAI_BASE.rstrip('/')}/responses"
    backoff = 1.5

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            # Raise for non-2xx early
            resp.raise_for_status()
            data = resp.json()
            text = extract_output_text(data)
            if not text:
                raise RuntimeError(f"Empty output from Responses API: {json.dumps(data)[:600]}")
            return text
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            logger.warning(f"[gpt] network timeout/conn error (attempt {attempt+1}/{retries+1}): {e}")
        except requests.HTTPError as e:
            last_err = e
            status = e.response.status_code if e.response is not None else None
            body_snip = ""
            try:
                body_snip = e.response.text[:300] if e.response is not None else ""
            except Exception:
                pass
            logger.warning(f"[gpt] HTTP {status} (attempt {attempt+1}/{retries+1}) {body_snip}")
            # Retry only transient statuses
            if status not in (429, 500, 502, 503, 504):
                break
        except Exception as e:
            last_err = e
            logger.warning(f"[gpt] unexpected error (attempt {attempt+1}/{retries+1}): {e}")

        # retry path
        if attempt < retries:
            time.sleep(backoff ** attempt)

    # Out of retries → raise the last error
    if last_err:
        raise last_err
    raise RuntimeError("Unknown GPT error")



# ---------- Personal bonus post-processor ----------
# NOTE: If your prompt no longer outputs a line like "3) Bonuses: ...",
# this post-processor will be a no-op. Keep or remove as you wish.
BONUS_POINTS = {
    "AI_LEADER": 30,
    "DIP_5_12": 50,
    "EARNINGS_BEAT": 25,
    "INSIDER_BUYING_30D": 20,
    "ANALYST_UPGRADES_7D": 15,
    "SECTOR_ROTATION_TAILWIND": 10,
    "BREAKOUT_VOL_CONF": 20,
    "SHORT_SQUEEZE_SETUP": 15,
    "INSTITUTIONAL_ACCUMULATION": 15,
    "POSITIVE_OPTIONS_SKEW": 10,
    "DIVIDEND_SAFETY_GROWTH": 5,
    "ESG_MOMENTUM": 5,
}

# Updated numbering to your latest 11-line format:
# 5) Final base score, 6) Personal adjusted score
RE_BASE = re.compile(r"^\s*5\)\s*Final base score:\s*(\d{1,4})\s*$", re.MULTILINE)
# If you *still* output "3) Bonuses: ..." keep this; otherwise function will do nothing.
RE_BONUS_LINE = re.compile(r"^\s*3\)\s*Bonuses:\s*(.+)$", re.MULTILINE)
RE_PERSONAL = re.compile(r"^\s*6\)\s*Personal adjusted score:\s*.*$", re.MULTILINE)

def _parse_bonus_flags(bonus_line: str):
    raw = bonus_line.strip()
    if raw.upper() == "NONE":
        return []
    flags = [x.strip().upper().replace("-", "_") for x in raw.split(",")]
    return [f for f in flags if f in BONUS_POINTS]

def apply_personal_bonuses_to_text(gpt_text: str) -> str:
    """
    Looks for a "3) Bonuses: ..." line in each block. If absent, leaves the block unchanged.
    Rewrites line 6) Personal adjusted score accordingly.
    """
    blocks = re.split(r"\n\s*\n", gpt_text.strip())
    new_blocks = []
    for b in blocks:
        base_m = RE_BASE.search(b)
        bonus_m = RE_BONUS_LINE.search(b)
        if not base_m or not bonus_m:
            new_blocks.append(b)
            continue
        try:
            base = int(base_m.group(1))
        except Exception:
            new_blocks.append(b)
            continue

        flags = _parse_bonus_flags(bonus_m.group(1))
        bonus_sum = sum(BONUS_POINTS[f] for f in flags)
        adjusted = base + bonus_sum
        breakdown = " + ".join([f"{BONUS_POINTS[f]} {f}" for f in flags]) if flags else "+0"
        personal_line = f"6) Personal adjusted score: {adjusted} (Base {base}{(' + ' + breakdown) if flags else ''})".rstrip()

        if RE_PERSONAL.search(b):
            b2 = RE_PERSONAL.sub(personal_line, b, count=1)
        else:
            lines = b.splitlines()
            out_lines, inserted = [], False
            for line in lines:
                out_lines.append(line)
                if RE_BASE.match(line) and not inserted:
                    out_lines.append(personal_line)
                    inserted = True
            b2 = "\n".join(out_lines)
        new_blocks.append(b2)
    return "\n\n".join(new_blocks)
