# scripts/gpt_client.py
import os
import time
import logging
from typing import Optional, Dict, Any, List

import requests

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("gpt_client")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("GPT_CLIENT_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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

    return None


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# -------- OpenAI call wrapper (Responses API) --------
def call_gpt5(
    system_msg: str,
    user_msg: str,
    *,
    model: str = None,
    max_tokens: int = None,
    timeout: Optional[float] = None,
    retries: int = 7,
) -> str:
    """
    Calls the OpenAI Responses API with retries + robust error handling.
    Optionally enables web search tool if OPENAI_ENABLE_WEB_SEARCH is true.
    """

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Default to the newest GPT available as of this code revision.
    model = model or os.getenv("OPENAI_MODEL", "gpt-5")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "10000")) if max_tokens is None else int(max_tokens)
    timeout = float(os.getenv("OPENAI_TIMEOUT", "360")) if timeout is None else float(timeout)

    enable_web = _bool_env("OPENAI_ENABLE_WEB_SEARCH", default=True)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_output_tokens": max_tokens,
        "text": {
            "format": {"type": "text"}
            }
                }

    # Enable web browsing (Responses API hosted tool)
    if enable_web:
        body["tools"] = [{"type": "web_search"}]
        # Ask for sources to be included in tool call output for logging/debugging.
        body["include"] = ["web_search_call.action.sources"]

    url = f"{OPENAI_BASE.rstrip('/')}/responses"
    backoff = 2.0

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            text = extract_output_text(data)
            if not text:
                logger.warning(f"[gpt] Empty response (attempt {attempt+1}/{retries}) — retrying...")
                time.sleep(backoff ** attempt)
                continue

            return text

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            logger.warning(f"[gpt] timeout/conn error (attempt {attempt+1}/{retries}): {e}")
        except requests.HTTPError as e:
            last_err = e
            status = e.response.status_code if e.response is not None else None
            snippet = ""
            try:
                snippet = (e.response.text or "")[:200]
            except Exception:
                pass
            logger.warning(f"[gpt] HTTP {status} (attempt {attempt+1}/{retries}) {snippet}")
            if status not in (429, 500, 502, 503, 504):
                break
        except Exception as e:
            last_err = e
            logger.warning(f"[gpt] unexpected error (attempt {attempt+1}/{retries}): {e}")

        if attempt < retries - 1:
            time.sleep(backoff ** attempt)

    if last_err:
        raise last_err
    raise RuntimeError("Unknown GPT error")