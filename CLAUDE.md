# Claude Code Guide (WhaleTraderBot)

This file is written for **Claude Code** to safely extend / fix this repo.

## What was added in this upgrade

1) **Progress UI** (Rich progress bars)
- Helper: `wtb/ui.py` (context manager `progress_context`)
- Used in: `wtb/pipeline.py` scan loops

2) **Scalp mode** (same pipeline, different config overrides)
- CLI: `python whaletraderbot.py scalp --side LONG`
- Overrides applied by: `wtb/pipeline.py::apply_mode_overrides()`
- Config block: `scalp.overrides` in `wtb/config.py` default_config

3) **Confidence gating** (EXECUTE/WATCH/SKIP)
- Heuristic scoring: `wtb/confidence.py::heuristic_confidence()`
- Integrated in: `wtb/pipeline.py` (filters SKIP from watchlist)
- Tune thresholds: `confidence.min_execute`, `confidence.min_watch`

4) **External intel (optional)** – provider interface + cache + aggregation
- Provider interface + built-ins: `wtb/external_intel_providers.py`
  - `MockIntelProvider` (neutral placeholders)
  - `HttpJSONProvider` (call your own bridge endpoints)
  - IMPORTANT: Providers must implement `name()`
- Cache + aggregation: `wtb/external_intel.py`
- Integrated in: `wtb/pipeline.py` (top-N candidates only)

---

## Repo entry points

- Main CLI: `whaletraderbot.py`
- Core scan pipeline: `wtb/pipeline.py`
- Manual mode: `wtb/manual.py`
- Backtest: `wtb/backtest.py`

---

## How to run (smoke)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Quick scan (no LLM calls)
python whaletraderbot.py scan --side LONG

# Scalp mode
python whaletraderbot.py scalp --side LONG

# Optional: print the ChatGPT audit prompt
python whaletraderbot.py scan --side LONG --print-prompt

# Run smoke checks
python -m wtb.tests_smoke
```

---

## TODOs (high value)

### TODO 1 — Add a real Grok / X / News provider

Current code ships with `mock` and `http` providers only.

Two safe approaches:

**A) HTTP bridge (recommended)**
- Keep Grok/X/news keys out of this repo.
- Create a tiny service (FastAPI, Cloudflare Worker, etc.) that exposes:
  - `GET {sentiment_url}?symbol=ETHUSDT&side=LONG`
  - `GET {onchain_url}?symbol=ETHUSDT&side=LONG`
  - `GET {events_url}?symbol=ETHUSDT&side=LONG`
- Return JSON shaped like the prompts in the README (or any dict; the aggregator expects `action_for_rrt_pb`).
- Configure:

```json
{
  "external_intel": {
    "enabled": true,
    "provider": "http",
    "http": {
      "sentiment_url": "https://YOUR_HOST/sentiment",
      "onchain_url": "https://YOUR_HOST/onchain",
      "events_url": "https://YOUR_HOST/events",
      "timeout_sec": 15
    }
  }
}
```

**B) Native provider inside repo**
- Add a new provider class in `wtb/external_intel_providers.py`.
- It MUST implement:
  - `name()`
  - `fetch_sentiment(symbol, context)`
  - `fetch_onchain(symbol, context)`
  - `fetch_events(symbol, context)`
- Avoid hardcoding secrets; read from env vars.

### TODO 2 — Replace heuristic confidence with a real ML model
- Keep the heuristic as a fallback.
- Add a training script that reads `outputs/*/payload.json` + realized trade outcomes.
- Candidate models:
  - LightGBM / XGBoost (binary: win >= 1R)
  - Calibrate with isotonic regression
- Add drift detection: reduce size or disable when feature distribution shifts.

### TODO 3 — Backtest realism upgrades
- Add fees, slippage, funding payment estimate
- Add intrabar simulation for 15m/5m scalp mode
- Add walk-forward + regime buckets (trend/range/high-vol)

---

## Bug-fix checklist

When something breaks, check in this order:

1) **Dependency import errors**
- `pip install -r requirements.txt`

2) **Binance API failures / rate limits**
- Retry/backoff around REST calls
- Reduce `scan.scan_top` or increase sleeps if needed

3) **Table UI / progress rendering issues**
- Disable progress via config:

```json
{ "ui": { "progress": false } }
```

4) **External intel shape issues**
- Ensure the provider returns JSON dicts.
- Ensure `action_for_rrt_pb` is one of:
  - `proceed`, `reduce_size_50`, `reduce_size_75`, `skip`, `pause_trading_24h`

5) **Confidence label not present**
- `wtb/pipeline.py` has safe defaults; but confirm `wtb/confidence.py` returns `label`.

---

## Coding conventions

- Keep deterministic ALGO outputs intact (Entry/SL/TP). LLMs must stay advisory.
- Prefer small pure functions. Avoid large new dependencies.
- Keep output JSON stable; version fields if you change schemas.

