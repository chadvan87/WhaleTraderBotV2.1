# WhaleTraderBot Strategy (Source-Based Summary)

This document summarizes the deterministic strategy and the step-by-step flow for each mode implemented in the current codebase. It is derived directly from source in `wtb/` and the CLI entrypoints.

## Strategy Core (Deterministic)

The system is a deterministic signal generator for USDT‑M perpetuals. All Entry, SL, and TP levels are computed by algorithmic rules and then refined by the level engine. Optional AI overlays (Plutus) and external intel are advisory only and do not change deterministic levels.

Key ideas:

- Trade mean‑reversion and inefficiency around ranges, breakouts, pullbacks, and volatility spikes.
- Use a non‑chasing, limit‑zone approach for entries.
- Align trades with BTC regime and market breath; penalize headwinds.
- Gate final decisions with a confidence heuristic to label EXECUTE/WATCH/SKIP.

## Deterministic Setup Types

Setup type is chosen by heuristics that look at ADX, ATR%, recent range structure, and recent candle behavior. The labels are:

- `TREND_PULLBACK`
- `BREAKOUT_RETEST`
- `RANGE_SWEEP_RECLAIM`
- `VOLATILITY_FADE`

Each setup has its own entry zone, SL, and TP logic. TPs are structure‑aware where possible, not purely R‑multiples.

## Level Engine Refinement

After the initial deterministic plan is built, levels are refined and validated using the level engine. It uses pivot clustering, AVWAP anchors, Bollinger metrics, and magnet levels, then applies validation and repairs when needed. If a repair occurs, `LEVELS_REPAIRED` is flagged but the plan remains deterministic.

## Scoring and Risk Controls

Final score is a weighted sum of component scores plus penalties. Components are tradeability, setup quality, derivatives, orderflow, context (BTC regime + market breath), and whales (Hyperliquid BTC/ETH context). Penalties include late entry, BTC headwind, and whale conflict. A separate confidence gate adjusts the decision using late status, derivatives flags, orderflow conflict, and optional external‑intel multipliers.

## Shared Pipeline Steps (Scan/Long/Short/Scalp/Manual)

These steps are common across scan and manual modes, with only the symbol source differing.

1. Load config and apply mode overrides. Scalp mode uses `scalp.overrides` if enabled.
2. Fetch BTC 1D klines and compute BTC regime (EMA50/EMA200, ATR%, ADX, volatility state).
3. Build the symbol universe (scan uses volume/spread filters, manual uses user input).
4. Compute 4H indicators (ATR/ADX) and a prescore for ranking.
5. Build deterministic ALGO plans per symbol with Entry/SL/TP.
6. Apply overlays as enabled: derivatives (funding/OI), orderflow (agg trades delta), whales, market breath, external intel, and Plutus psychology overlay. Overlays do not alter Entry/SL/TP.
7. Compute final scores with penalties.
8. Run confidence gate to label EXECUTE/WATCH/SKIP.
9. Select watchlist and write outputs.

Outputs (default `outputs/latest/`):

- `payload.json`
- `watchlist.txt`
- `shortlist_table.txt`
- `chatgpt_prompt.txt`

## Mode Details

### Scan Mode (`scan`, `long`, `short`)

Source: `wtb/pipeline.py`, CLI in `wtb/cli.py`

Steps:

1. Pull Binance Futures tickers and book tickers.
2. Filter to USDT‑M perps, sort by 24h quote volume.
3. Apply soft filters for minimum volume and maximum spread.
4. Pre‑score by volume, spread, ATR%, and ADX, then keep top `shortlist_n`.
5. Build algo plans with deterministic Entry/SL/TP, then refine via level engine.
6. Optional overlays can be enabled or disabled by config or CLI flags.
7. Score + penalties.
8. Confidence gate assigns EXECUTE/WATCH/SKIP.
9. Watchlist selection uses `min_watch_score` and `watchlist_k`. In RISK_OFF breath, it raises min score and reduces watchlist size.

### Scalp Mode (`scalp`)

Source: `wtb/pipeline.py` via `apply_mode_overrides`

Scalp mode is the same pipeline as scan but uses config overrides (faster timeframes and tighter thresholds) defined in `config.json` under `scalp.overrides`.

### Manual Mode (`manual`)

Source: `wtb/manual.py`

Differences from scan mode:

- Symbols come from user input or `manual.default_symbols` in config.
- Symbols are validated against Binance USDT‑M perps.
- No `shortlist_n` cap; all valid symbols are fully analyzed.
- Orderflow overlay runs for all symbols (not just top‑N).
- Output `payload.json` includes `all_plans` for full transparency.

### Backtest Mode (`backtest`)

Source: `wtb/backtest.py`

Walk‑forward backtest that replays the same deterministic algo on historical klines.

Steps:

1. Download historical klines for the symbol and BTC 1D for regime.
2. Walk bar‑by‑bar using a rolling lookback window.
3. Build algo plans at each signal point.
4. Enforce cooldown between signals.
5. Wait for fill within `fill_window` bars.
6. Manage exits via SL, TP1/TP2/TP3, or timeout.
7. Compute performance stats (win rate, R‑multiple, profit factor, drawdown).

### Cache Mode (`cache`)

Source: `wtb/backtest.py`

Utility to download and store historical klines to Parquet for offline analysis.

### DCA Discovery Mode (`dca`)

Source: `wtb/dca.py`, `wtb/dca_scoring.py`

This is a separate pipeline focused on DCA/grid suitability, not the main ALGO watchlist.

Steps:

1. Pull Binance tickers and book tickers, build the DCA universe (top‑N by volume).
2. Assign each symbol to `CORE`, `MID`, or `EXPLORE` by volume/spread thresholds.
3. Compute DCA score from five components: microstructure, mean reversion, volatility fit, derivatives health, and context (BTC regime).
4. Apply penalties such as BTC headwind, extreme funding, liquidity stress, and trend runaway.
5. Compute deterministic execution plan (Entry/SL/TPs) and validate levels.
6. Suggest DCA profile (grid step %, max layers, size multiplier).
7. Generate kill‑switch conditions based on ATR and tier.
8. Apply explore‑quota cap and min score to build the final list.

Outputs (default `outputs/latest/`):

- `dca_payload.json`
- `dca_watchlist.txt`
- `dca_chatgpt_prompt.txt`

## Decision Labels

Confidence gating labels each candidate:

- `EXECUTE`: score >= `confidence.min_execute`
- `WATCH`: score >= `confidence.min_watch`
- `SKIP`: below watch threshold or external‑intel veto

## Determinism and Overlays

- Deterministic: Entry, SL, TPs, and all core scoring.
- Advisory only: Plutus (Ollama psychology) and external intel providers.
- Overlays do not modify Entry/SL/TP or core algorithm outputs.

