# scripts/debugger.py
import requests
from typing import List, Dict, Any

# Discord limits we care about
_MAX_EMBEDS_PER_MESSAGE = 9          # hard cap is 10; keep 1 buffer
_MAX_FIELD_VALUE = 3754              # per field value
_MAX_TITLE_LEN = 256

def _fmt_num(x):
    try:
        return "N/A" if x is None else f"{float(x):.2f}"
    except Exception:
        return "N/A"

def _fmt_pct(x):
    try:
        return "N/A" if x is None else f"{float(x):.2f}%"
    except Exception:
        return "N/A"

def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")

def post_debug_inputs_to_discord(picked: List[str], debug_inputs: Dict[str, Dict[str, Any]], webhook_url: str):
    """
    picked: list[str] of tickers GPT selected (or fallback set)
    debug_inputs: {ticker: debug_dict} (as built in prompt_blocks.build_prompt_block)
    webhook_url: Discord webhook (can be same as alert)
    """
    if not picked or not webhook_url:
        return

    embeds: List[Dict[str, Any]] = []
    for t in picked:
        d = debug_inputs.get(t) or {}
        vals = d.get("VALUATION_FIELDS_DICT") or {}
        prox = (d.get("PROXIES_BLOCK") or {})
        ind  = (d.get("INDICATORS_BLOCK") or {})

        title_raw = f"DEBUG — {t} ({d.get('COMPANY','?')})"
        title = _truncate(title_raw, _MAX_TITLE_LEN)

        desc_raw  = f"Price: {d.get('CURRENT_PRICE','N/A')}  |  FVA_HINT: {_fmt_num((prox.get('FVA_HINT') if isinstance(prox, dict) else None))}"
        desc = _truncate(desc_raw, 2048)  # Discord description limit

        # Indicators summary
        indicators_text = (
            f"RSI14 {_fmt_num(ind.get('RSI14'))} · ATR% {_fmt_num(ind.get('ATR_pct'))} · "
            f"vsSMA50 {_fmt_num(ind.get('vsSMA50_pct'))}% · vsSMA200 {_fmt_num(ind.get('vsSMA200_pct'))}% · "
            f"Vol_vs_20d {_fmt_num(ind.get('Vol_vs_20d_pct'))}%"
        )
        indicators_text = _truncate(indicators_text, _MAX_FIELD_VALUE)

        # Proxies summary
        proxies_text = (
            f"MT {prox.get('MARKET_TREND','?')} | RS {prox.get('REL_STRENGTH','?')} | BV {prox.get('BREADTH_VOLUME','?')} | "
            f"ValHist {prox.get('VALUATION_HISTORY','?')} | RiskVol {prox.get('RISK_VOLATILITY','?')} | RiskDD {prox.get('RISK_DRAWDOWN','?')}"
        )
        proxies_text = _truncate(proxies_text, _MAX_FIELD_VALUE)

        # Valuations
        valuations_text = (
            f"PE: {_fmt_num(vals.get('PE'))} | PEG: {_fmt_num(vals.get('PEG'))}\n"
            f"FCF Yield: {_fmt_pct(vals.get('FCF_YIELD_pct') if 'FCF_YIELD_pct' in vals else vals.get('FCF_YIELD'))}\n"
            f"EV/EBITDA: {_fmt_num(vals.get('EV_EBITDA'))} | EV/Rev: {_fmt_num(vals.get('EV_REV'))} | P/S: {_fmt_num(vals.get('PS'))}"
        )
        valuations_text = _truncate(valuations_text, _MAX_FIELD_VALUE)

        # Prompt preview (safe length; embed field value <= 1024)
        prompt_block = (d.get("PROMPT_BLOCK") or "").strip()
        if len(prompt_block) > 900:
            prompt_block = prompt_block[:3600] + " …(truncated)"
        prompt_field_val = f"```text\n{prompt_block}\n```"
        prompt_field_val = _truncate(prompt_field_val, _MAX_FIELD_VALUE)

        embed = {
            "title": title,
            "description": desc,
            "fields": [
                {"name": "Indicators", "value": indicators_text or "N/A", "inline": False},
                {"name": "Proxies", "value": proxies_text or "N/A", "inline": False},
                {"name": "Valuations", "value": valuations_text or "N/A", "inline": False},
                {"name": "Prompt Block (preview)", "value": prompt_field_val or "N/A", "inline": False},
            ],
            "color": 0x4B9CD3,
        }
        embeds.append(embed)

    # Send in chunks to respect Discord's embed-per-message limits
    for i in range(0, len(embeds), _MAX_EMBEDS_PER_MESSAGE):
        chunk = embeds[i:i + _MAX_EMBEDS_PER_MESSAGE]
        try:
            resp = requests.post(
                webhook_url,
                json={"username": "Daily Stock Debug", "embeds": chunk},
                timeout=30
            )
            if not (200 <= resp.status_code < 300):
                print(f"[WARN] Debug Discord post failed: HTTP {resp.status_code} {resp.text}", flush=True)
        except Exception as e:
            # Don’t crash the main flow if debug post fails
            print(f"[WARN] Debug Discord post failed: {e}", flush=True)
