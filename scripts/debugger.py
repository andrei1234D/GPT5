# scripts/debugger.py
import requests
from typing import List, Dict, Any

# Discord limits
_MAX_EMBEDS_PER_MESSAGE = 9          # safe (Discord hard cap is 10)
_MAX_TITLE_LEN = 256
_MAX_DESC_LEN = 2048
_MAX_FIELD_NAME_LEN = 256
_MAX_FIELD_VALUE_LEN = 1024
_EMBED_CHAR_BUDGET = 5900            # stay under ~6000 safety margin
_MAX_FIELDS_PER_EMBED = 25           # Discord cap

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

def _chunk(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i:i+n]

def _embed_len(e: Dict[str, Any]) -> int:
    # Approximate char count used in length validation
    size = len(e.get("title", "") or "") + len(e.get("description", "") or "")
    for f in e.get("fields", []) or []:
        size += len(f.get("name", "") or "") + len(f.get("value", "") or "")
    if "footer" in e and isinstance(e["footer"], dict):
        size += len(e["footer"].get("text", "") or "")
    if "author" in e and isinstance(e["author"], dict):
        size += len(e["author"].get("name", "") or "")
    return size

def _add_field_safe(embed: Dict[str, Any], name: str, value: str, inline: bool=False) -> bool:
    fields = embed.setdefault("fields", [])
    if len(fields) >= _MAX_FIELDS_PER_EMBED:
        return False
    field = {
        "name": _truncate(name, _MAX_FIELD_NAME_LEN) or "\u200b",
        "value": _truncate(value, _MAX_FIELD_VALUE_LEN) or "\u200b",
        "inline": inline,
    }
    fields.append(field)
    # If over budget, revert and fail
    if _embed_len(embed) > _EMBED_CHAR_BUDGET:
        fields.pop()
        return False
    return True

def _attach_file(webhook_url: str, ticker: str, company: str, prompt_block: str):
    try:
        fname = f"debug_{ticker}.txt"
        content = f"DEBUG — {ticker} ({company})\n\n{prompt_block or ''}\n"
        files = {"file": (fname, content.encode("utf-8"), "text/plain")}
        resp = requests.post(
            webhook_url,
            data={"username": "Daily Stock Debug", "content": f"Full prompt for **{ticker}** attached."},
            files=files,
            timeout=30
        )
        if not (200 <= resp.status_code < 300):
            print(f"[WARN] Debug Discord file post failed: HTTP {resp.status_code} {resp.text}", flush=True)
    except Exception as e:
        print(f"[WARN] Debug Discord file post failed: {e}", flush=True)

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
        company = d.get("COMPANY", "?")

        title = _truncate(f"DEBUG — {t} ({company})", _MAX_TITLE_LEN)
        desc_raw = f"Price: {d.get('CURRENT_PRICE','N/A')}  |  FVA_HINT: {_fmt_num((prox.get('FVA_HINT') if isinstance(prox, dict) else None))}"
        desc = _truncate(desc_raw, _MAX_DESC_LEN)

        embed = {"title": title, "description": desc, "color": 0x4B9CD3, "fields": []}

        # Indicators summary (keep compact)
        indicators_text = (
            f"RSI14 {_fmt_num(ind.get('RSI14'))} · ATR% {_fmt_num(ind.get('ATR_pct'))} · "
            f"vsSMA50 {_fmt_num(ind.get('vsSMA50_pct'))}% · vsSMA200 {_fmt_num(ind.get('vsSMA200_pct'))}% · "
            f"Vol_vs_20d {_fmt_num(ind.get('Vol_vs_20d_pct'))}%"
        )
        _add_field_safe(embed, "Indicators", indicators_text, inline=False)

        # Proxies summary
        proxies_text = (
            f"MT {prox.get('MARKET_TREND','?')} | RS {prox.get('REL_STRENGTH','?')} | BV {prox.get('BREADTH_VOLUME','?')} | "
            f"ValHist {prox.get('VALUATION_HISTORY','?')} | RiskVol {prox.get('RISK_VOLATILITY','?')} | RiskDD {prox.get('RISK_DRAWDOWN','?')}"
        )
        _add_field_safe(embed, "Proxies", proxies_text, inline=False)

        # Valuations
        valuations_text = (
            f"PE: {_fmt_num(vals.get('PE'))} | PEG: {_fmt_num(vals.get('PEG'))}\n"
            f"FCF Yield: {_fmt_pct(vals.get('FCF_YIELD_pct') if 'FCF_YIELD_pct' in vals else vals.get('FCF_YIELD'))}\n"
            f"EV/EBITDA: {_fmt_num(vals.get('EV_EBITDA'))} | EV/Rev: {_fmt_num(vals.get('EV_REV'))} | P/S: {_fmt_num(vals.get('PS'))}"
        )
        _add_field_safe(embed, "Valuations", valuations_text, inline=False)

        # Prompt preview – chunk into multiple fields if needed
        prompt_block = (d.get("PROMPT_BLOCK") or "").strip()
        attached = False
        if prompt_block:
            # Aim for chunks that fit under field value limit incl. code fences & header
            CHUNK = 900
            parts = list(_chunk(prompt_block, CHUNK))
            added_any = False
            for idx, chunk in enumerate(parts, start=1):
                header = f"Prompt ({idx}/{len(parts)})"
                body = f"```text\n{chunk}\n```"
                ok = _add_field_safe(embed, header, body, inline=False)
                if not ok:
                    # Try fewer parts: attach file if we can't add more without overflow
                    if not added_any:
                        # no prompt content fit → attach full file
                        _attach_file(webhook_url, t, company, prompt_block)
                        attached = True
                    else:
                        # some parts added but would overflow now → attach remainder
                        rest = "".join(parts[idx-1:])  # include current chunk and the rest
                        _attach_file(webhook_url, t, company, rest)
                        attached = True
                    break
                else:
                    added_any = True

            if attached:
                _add_field_safe(embed, "Prompt Block (note)", "Prompt too long for embed; full text attached as file.", inline=False)

        # Final sanity: if somehow we still exceed budget, drop last fields until valid
        while _embed_len(embed) > _EMBED_CHAR_BUDGET and embed.get("fields"):
            embed["fields"].pop()

        embeds.append(embed)

    # Send in chunks to respect Discord's per-message embed limit
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
            print(f"[WARN] Debug Discord post failed: {e}", flush=True)