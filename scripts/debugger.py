# scripts/debugger.py
import json
import requests

def post_debug_inputs_to_discord(tickers: list[str], debug_inputs: dict[str, dict], webhook_url: str):
    """Send JSON dumps of the exact GPT inputs per ticker (chunked to avoid embed limits)."""
    for t in tickers:
        data = debug_inputs.get(t)
        if not data:
            continue
        try:
            blob = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            safe = {k: (str(v) if not isinstance(v, (str, int, float, bool, dict, list)) else v)
                    for k, v in (data or {}).items()}
            blob = json.dumps(safe, indent=2, ensure_ascii=False)

        # Discord embed description limit ~4096; keep chunks around 3400–3500
        chunk_size = 3400
        chunks = [blob[i:i+chunk_size] for i in range(0, len(blob), chunk_size)] or ["{}"]

        for idx, ch in enumerate(chunks, start=1):
            title = f"Model inputs — {t}" + (f" ({idx}/{len(chunks)})" if len(chunks) > 1 else "")
            payload = {
                "username": "Daily Stock Debug",
                "embeds": [{
                    "title": title,
                    "description": f"```json\n{ch}\n```"
                }]
            }
            try:
                requests.post(webhook_url, json=payload, timeout=60).raise_for_status()
            except Exception as e:
                print(f"[WARN] Debug Discord post failed for {t} part {idx}: {repr(e)}", flush=True)
