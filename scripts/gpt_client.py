# scripts/gpt_client.py
import os
import re
import requests

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_output_text(data: dict) -> str | None:
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "").strip()
    return None

def call_gpt5(system_msg: str, user_msg: str, max_tokens: int = 7000) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-5",
        "input": f"[System]: {system_msg}\n[User]: {user_msg}",
        "max_output_tokens": max_tokens,
        "temperature": 1.0
    }
    r = requests.post(f"{OPENAI_BASE}/responses", headers=headers, json=body, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = extract_output_text(data)
    if not text:
        raise RuntimeError(f"Empty output from Responses API: {str(data)[:400]}")
    return text

# ---------- Personal bonus post-processor ----------
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

RE_BASE = re.compile(r"^\s*6\)\s*Final base score:\s*(\d{1,4})\s*$", re.MULTILINE)
RE_BONUS_LINE = re.compile(r"^\s*3\)\s*Bonuses:\s*(.+)$", re.MULTILINE)
RE_PERSONAL = re.compile(r"^\s*7\)\s*Personal adjusted score:\s*.*$", re.MULTILINE)

def _parse_bonus_flags(bonus_line: str):
    raw = bonus_line.strip()
    if raw.upper() == "NONE": return []
    flags = [x.strip().upper().replace("-", "_") for x in raw.split(",")]
    return [f for f in flags if f in BONUS_POINTS]

def apply_personal_bonuses_to_text(gpt_text: str) -> str:
    blocks = re.split(r"\n\s*\n", gpt_text.strip())
    new_blocks = []
    for b in blocks:
        base_m = RE_BASE.search(b); bonus_m = RE_BONUS_LINE.search(b)
        if not base_m or not bonus_m:
            new_blocks.append(b); continue
        try:
            base = int(base_m.group(1))
        except Exception:
            new_blocks.append(b); continue

        flags = _parse_bonus_flags(bonus_m.group(1))
        bonus_sum = sum(BONUS_POINTS[f] for f in flags)
        adjusted = base + bonus_sum
        breakdown = " + ".join([f"{BONUS_POINTS[f]} {f}" for f in flags]) if flags else "+0"
        personal_line = f"7) Personal adjusted score: {adjusted} (Base {base}{(' + ' + breakdown) if flags else ''})".rstrip()

        if RE_PERSONAL.search(b):
            b2 = RE_PERSONAL.sub(personal_line, b, count=1)
        else:
            lines = b.splitlines()
            out_lines, inserted = [], False
            for line in lines:
                out_lines.append(line)
                if RE_BASE.match(line) and not inserted:
                    out_lines.append(personal_line); inserted = True
            b2 = "\n".join(out_lines)
        new_blocks.append(b2)
    return "\n\n".join(new_blocks)
