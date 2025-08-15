# scripts/debugger.py
import requests
import json

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

def post_debug_inputs_to_discord(picked, debug_inputs, webhook_url):
    """
    picked: list[str] of tickers GPT selected
    debug_inputs: {ticker: debug_dict} (as built in prompt_blocks.build_prompt_block)
    webhook_url: Discord webhook
    """
    if not picked:
        return

    embeds = []
    for t in picked:
        d = debug_inputs.get(t) or {}
        vals = d.get("VALUATION_FIELDS_DICT") or {}
        prox = (d.get("PROXIES_BLOCK") or {})
        ind  = (d.get("INDICATORS_BLOCK") or {})

        title = f"DEBUG — {t} ({d.get('COMPANY','?')})"
        desc  = f"Price: {d.get('CURRENT_PRICE','N/A')}  |  FVA_HINT: {_fmt_num(prox.get('FVA_HINT'))}"

        # --- Indicators summary (short) ---
        indicators_text = (
            f"RSI14 {_fmt_num(ind.get('RSI14'))} · ATR% {_fmt_num(ind.get('ATR_pct'))} · "
            f"vsSMA50 {_fmt_num(ind.get('vsSMA50_pct'))}% · vsSMA200 {_fmt_num(ind.get('vsSMA200_pct'))}% · "
            f"Vol_vs_20d {_fmt_num(ind.get('Vol_vs_20d_pct'))}%"
        )

        # --- Proxies summary (short) ---
        proxies_text = (
            f"MT {prox.get('MARKET_TREND','?')} | RS {prox.get('REL_STRENGTH','?')} | BV {prox.get('BREADTH_VOLUME','?')} | "
            f"ValHist {prox.get('VALUATION_HISTORY','?')} | RiskVol {prox.get('RISK_VOLATILITY','?')} | RiskDD {prox.get('RISK_DRAWDOWN','?')}"
        )

        # --- Valuations section (this is the new bit you asked for) ---
        valuations_text = (
            f"PE: {_fmt_num(vals.get('PE'))} | PEG: {_fmt_num(vals.get('PEG'))}\n"
            f"FCF Yield: {_fmt_pct(vals.get('FCF_YIELD_pct'))}\n"
            f"EV/EBITDA: {_fmt_num(vals.get('EV_EBITDA'))} | EV/Rev: {_fmt_num(vals.get('EV_REV'))} | P/S: {_fmt_num(vals.get('PS'))}"
        )

        # PROMPT block (trim so we don’t blow Discord limits)
        prompt_block = (d.get("PROMPT_BLOCK") or "").strip()
        if len(prompt_block) > 1000:
            prompt_block = prompt_block[:1000] + " …(truncated)"

        embed = {
            "title": title,
            "description": desc,
            "fields": [
                {"name": "Indicators", "value": indicators_text or "N/A", "inline": False},
                {"name": "Proxies", "value": proxies_text or "N/A", "inline": False},
                {"name": "Valuations", "value": valuations_text or "N/A", "inline": False},
                {"name": "Prompt Block (preview)", "value": f"```text\n{prompt_block}\n```", "inline": False},
            ],
            "color": 0x4B9CD3,  # nice blue
        }
        embeds.append(embed)

    try:
        requests.post(webhook_url, json={"username": "Daily Stock Debug", "embeds": embeds}, timeout=30).raise_for_status()
    except Exception as e:
        # Don’t crash the main flow if debug post fails
        print(f"[WARN] Debug Discord post failed: {e}", flush=True)
