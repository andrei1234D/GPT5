# GPT5 Codebase Instructions

## Project Overview
GPT5 is an **automated stock screening and daily alert system** that:
1. Filters a universe of stocks from Trading 212 via `scripts/` pipeline
2. Scores candidates through technical analysis + LLM embeddings  
3. Runs a locally-trained Hugging Face regression model ("Brain") to rank candidates
4. Sends top-10 picks to Discord via GPT-5-analyzed commentary

**Key pattern**: Multi-stage filtering → technical scoring → LLM-based re-ranking → final LLM narrative.

---

## Architecture & Data Flow

### Pipeline Stages (see `.github/workflows/discord-stock-alert.yml`)
```
1. prepare-data job:
   universe_from_trading_212.py → (raw universe)
   clean_universe.py → (universe_clean.csv)
   aliases_autobuild.py → (aliases.csv for unmapped symbols)
   quick_scorer.py → (stage1_kept.csv, top 200 by quality)
   trash_ranker.py → (stage2_merged.csv, final 10–20 candidates)
   llm_data_builder.py → (LLM_today_data.jsonl with full feature set)
   brain_ranker.py → (rank via PyTorch model)

2. send-alert job (after prepare-data completes):
   notify.py → reads top-10 Brain ranks, calls GPT-5 with formatted CSV, posts to Discord
```

### Key Files
- **Universe building**: `scripts/universe_from_trading_212.py`, `scripts/clean_universe.py`
- **Technical scoring**: `scripts/quick_scorer.py` (Stage 1), `scripts/trash_ranker.py` (Stage 2)
- **LLM pipeline**: `scripts/llm_data_builder.py` (feature extraction), `scripts/brain_ranker.py` (PyTorch ranking)
- **GPT integration**: `scripts/gpt_client.py` (Responses API retry logic), `scripts/notify.py` (orchestration)
- **Technical features**: `scripts/features.py` (EMA, RSI, MACD, volatility), `scripts/data_fetcher.py` (yfinance caching)
- **Prompts**: `scripts/prompts.py` (system/user message templates for GPT-5)

---

## Critical Patterns & Conventions

### 1. **Environment Variables** (NOT in code, in Actions)
Heavy use of env-driven configuration:
```bash
# Stage 1 (Quick Scorer)
STAGE1_MODE='loose'  # or 'quick' (affects filtering strictness)
STAGE1_KEEP='200'    # number of candidates to keep
QS_PE_WEIGHT='0.015' # valuation overlay weighting

# Stage 2 (Trash Ranker)
RANKER_PROFILE='C'   # profile for scoring rules
EARLY_TURN_MIN_E50_SLOPE='2.0'  # trend confirmation threshold

# LLM/Brain
BRAIN_MODEL_DIR='scripts/train_model/LLM_bot/Brain/llm_signal_regression_model'
OPENAI_MODEL='gpt-5'
OPENAI_MAX_TOKENS='15000'

# Feature extraction
YF_CHUNK_SIZE='50'  # batch download size
YF_MAX_RETRIES='5'
YF_SLEEP_PER_REQUEST='0.5'  # rate limiting
```
→ **Pattern**: When adding features, add an env var to control it, don't hardcode.

### 2. **Symbol Normalization**
- Consistent across `features.py`, `data_fetcher.py`, `llm_data_builder.py`:
- International symbols use Yahoo Finance suffixes (e.g., `ASML.AS` for Amsterdam)
- Alias system in `scripts/aliases.py` maps malformed tickers to clean ones
- Clean universe stored in `data/universe_clean.csv` (ticker + company name)
- Rejects stored in `data/universe_rejects.csv` (ticker + reason, for next run's alias-building)

### 3. **Data Dependencies & File Caching**
```
data/universe.csv → data/universe_clean.csv  (human-curated input → cleaned output)
                 → data/universe_rejects.csv (persistent: used by aliases_autobuild)
                 
data/stage1_kept.csv → data/stage2_merged.csv (Quick Scorer output → Trash Ranker input)
                    → logs/top2000_quick_full.csv (diagnostic: top-N scores)

data/LLM_today_data.jsonl → Brain Model ranks (all features + Brain scores)
                         → notify.py consumes (picks top-10, fetches from stage2, merges news)
```
→ **Pattern**: Each stage commits outputs to git (with commit messages referencing run #). Always check git state before modifying.

### 4. **Brain Model Loading & Git LFS**
- Located at `scripts/train_model/LLM_bot/Brain/llm_signal_regression_model/model.safetensors`
- Stored via Git LFS; workflow uses `lfs: true` in checkout
- `brain_ranker.py` checks file size: if <1 MB, assumes LFS pointer error and raises `RuntimeError`
- **Pattern**: Always call `_load_model()` before inference; it validates LFS state

### 5. **LLM Data Builder → Feature Parity**
`llm_data_builder.py` rebuilds features for inference:
- Must match training dataset structure (stored in git history)
- Computes: SMA20/50/200, EMA20/50/200, RSI14, MACD, ATR%, Volatility, Momentum, OBV
- Handles yfinance MultiIndex flattening (when downloading multiple symbols at once)
- Returns single latest record per ticker in JSONL format
- **Pattern**: If you add a feature to Brain training, add it here too (with comments showing training source)

### 6. **GPT-5 Responses API** (OpenAI)
- Custom Responses API (not standard Chat API)
- `gpt_client.py`:
  - Retries with exponential backoff (base 2.0, up to 7 attempts)
  - Retries only on timeout, connection error, or 5xx/429; **NOT** on permanent errors
  - Extracts text from 3 possible API response shapes (robust for SDK variations)
  - Bonus post-processor (`apply_personal_bonuses_to_text`) parses "3) Bonuses: ..." and calculates adjusted scores

### 7. **Prompts & Scoring Format**
- System prompt: `SYSTEM_PROMPT_TOP20` (~300 lines, stored in `scripts/prompts.py`)
- User prompt: `USER_PROMPT_TOP20_TEMPLATE` (formatted with date + CSV blocks)
- CSV passed to GPT: minimal (ticker, price, RSI14, MACD_hist, Momentum, ATRpct, volatility_30, pos_30d, EMA50, EMA200, MarketTrend, News)
- **Pattern**: CSV is created in-memory, written to `logs/blocks_to_gpt_latest.txt` for audit trail
- GPT outputs structured blocks per ticker (1-indexed components → final base score → adjusted score)

### 8. **Discord Integration**
- Webhook-based: embeds formatted JSON to Discord
- Fallback webhook: `DISCORD_DEBUG_WEBHOOK_URL` (if set, errors go here too)
- On any pipeline failure: sends error to webhook via `requests.post()` with timeout=60
- **Pattern**: No retries on webhook failures; logged + continue

### 9. **Robust Float Handling**
Across all feature code (e.g., `data_fetcher.py`, `llm_data_builder.py`):
```python
def _safe_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except: return None
```
→ **Pattern**: All NaN/Inf values become `None` during serialization; treated as missing in downstream logic.

### 10. **Logging & Debugging**
- Each module uses `logging.basicConfig()` if not already configured
- Env-driven log levels: `FEATURES_LOG_LEVEL`, `RANKER_LOG_LEVEL`, `GPT_CLIENT_LOG_LEVEL`
- Logs periodically (e.g., every 300 requests in features.py: `FEATURES_LOG_EVERY=300`)
- Timestamps in Europe/Bucharest timezone (`TZ = pytz.timezone("Europe/Bucharest")`)
- **Pattern**: Use `flush=True` in print() for immediate GitHub Actions visibility

---

## Common Modifications

### Adding a New Technical Feature
1. Add computation to `scripts/features.py` (in `compute_indicators()` or similar)
2. Add to LLM data builder: `scripts/llm_data_builder.py` in `_build_single_record()`
3. If used in scoring: add env var to workflows, update Quick Scorer or Trash Ranker
4. Update prompts if GPT-5 should see it: modify CSV header in `scripts/notify.py`

### Adjusting Scoring Weights
- Stage 1 (Quick Scorer): env vars like `QS_PE_WEIGHT`, `QS_ET_BONUS_TREND`
- Stage 2 (Trash Ranker): env vars like `RANKER_PROFILE`, `EARLY_TURN_MIN_E50_SLOPE`
- **Pattern**: Update workflow `.yml` only; no code changes needed

### Retraining Brain Model
- Training data: prepare in `scripts/train_model/generate_training_data.py`
- Model: saved to `scripts/train_model/LLM_bot/Brain/llm_signal_regression_model/`
- Use `safetensors` format (not pickle)
- After retraining: commit to git with LFS; workflow will fetch it

### Testing Changes Locally
- Prepare a recent `data/LLM_today_data.jsonl` from main branch
- Set env vars, then run: `python scripts/notify.py` (or individual stages)
- Check `daily_pick.txt` output and `logs/blocks_to_gpt_latest.txt` for GPT input audit trail

---

## Testing & Validation

### Unit Tests
- Limited test coverage (see `scripts/test_brain_ranker.py`)
- Main validation: workflow runs + git history
- Manual: download recent LLM data, run `brain_ranker.py` locally, check scores are reasonable

### Workflow Validation
- GitHub Actions runs daily at 04:00 UTC
- Can trigger manually via `workflow_dispatch` (workflow page)
- Check: all stages complete → outputs committed → Discord alert posted
- Failures: review GitHub Actions logs for stage that failed (common: yfinance rate-limit, LFS not fetched)

---

## Performance & Scaling Notes

- **yfinance rate-limiting**: backoff env vars (`YF_SLEEP_PER_REQUEST`, `YF_RETRY_SLEEP`)
- **Brain model**: runs in batches (default `batch_size=8`), GPU auto-detected
- **GPT-5 calls**: single call per run, timeout=360s, retries=7
- **Discord webhook**: timeout=60s, no retry
- **Caching**: yfinance data cached locally during run; valuation caches TTL'd (1 day) and purged before pipeline

---

## Key Dependencies
- `pandas`, `numpy` (data manipulation)
- `yfinance` (market data)
- `requests` (webhooks, OpenAI API)
- `torch`, `transformers`, `safetensors` (Brain model inference)
- `pytz` (timezone handling)

---

## Common Issues & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Brain model not found | LFS not checked out | Ensure `lfs: true` in Actions checkout |
| "Model file looks like Git LFS pointer" | LFS file too small | Re-fetch repo with `git lfs pull` locally |
| yfinance timeout / rate-limit | Too many requests | Increase `YF_SLEEP_PER_REQUEST`, reduce `YF_CHUNK_SIZE` |
| GPT-5 returns empty | Responses API format changed | Check `extract_output_text()` handles new shape |
| Discord webhook fails | Wrong webhook URL | Verify `DISCORD_WEBHOOK_URL` secret in Actions |
| News summary missing | AlphaVantage rate-limited | Check `ALPHAVANTAGE_API_KEY`, reduce news fetches |

