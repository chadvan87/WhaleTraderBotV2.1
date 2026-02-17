from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .binance import BinanceFuturesClient
from .algo import build_algo_plan, AlgoPlan
from .indicators import atr, adx, ema
from .regime import detect_btc_regime


console = Console()


# ---------------------------------------------------------------------------
# Parquet caching (original functionality)
# ---------------------------------------------------------------------------

def cache_klines(client: BinanceFuturesClient, symbol: str, interval: str, start_ms: int, end_ms: int, out_path: pathlib.Path) -> pathlib.Path:
    """Download klines into a Parquet cache (offline backtest support)."""
    rows: List[List] = []
    cursor = start_ms
    step = 1500  # max limit per Binance API call
    while True:
        batch = client.klines(symbol, interval, limit=step, start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open == cursor:
            break
        cursor = last_open + 1
        if cursor >= end_ms:
            break
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Backtest engine — replays WhaleTraderBot algo signals on historical klines
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed or open trade."""
    bar_index: int           # bar where signal was generated
    fill_index: int          # bar where entry was filled (-1 if never filled)
    exit_index: int          # bar where trade was closed (-1 if still open)
    symbol: str
    side: str
    setup_type: str
    entry_low: float
    entry_high: float
    fill_price: float        # actual fill price (mid of zone touch)
    stop_loss: float
    take_profits: List[float]
    tp_hit: int              # which TP was hit (0=none, 1/2/3)
    exit_price: float
    exit_reason: str         # "TP1", "TP2", "TP3", "SL", "TIMEOUT", "OPEN"
    pnl_pct: float           # percentage PnL (before fees)
    r_multiple: float        # realized R-multiple
    algo_score: float        # algo score_setup at signal time


@dataclass
class BacktestResult:
    """Aggregated backtest statistics."""
    symbol: str
    side: str
    interval: str
    total_bars: int
    signals_generated: int
    trades_filled: int
    trades_won: int
    trades_lost: int
    trades_breakeven: int
    trades_timeout: int
    win_rate: float
    avg_r: float
    total_r: float
    max_r: float
    min_r: float
    profit_factor: float
    avg_pnl_pct: float
    total_pnl_pct: float
    max_drawdown_r: float
    trades: List[Trade]
    equity_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "interval": self.interval,
            "total_bars": self.total_bars,
            "signals_generated": self.signals_generated,
            "trades_filled": self.trades_filled,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "trades_breakeven": self.trades_breakeven,
            "trades_timeout": self.trades_timeout,
            "win_rate": round(self.win_rate, 4),
            "avg_r": round(self.avg_r, 4),
            "total_r": round(self.total_r, 4),
            "max_r": round(self.max_r, 4),
            "min_r": round(self.min_r, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "max_drawdown_r": round(self.max_drawdown_r, 4),
            "num_trades": len(self.trades),
        }


def _parse_entry_zone(zone_str: str) -> Tuple[float, float]:
    """Parse 'low-high' entry zone string from AlgoPlan."""
    parts = zone_str.split("-")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    # Fallback: sometimes scientific notation uses '-' for negative exponents
    # Try to find the separator dash that isn't part of a number
    for i in range(1, len(zone_str)):
        if zone_str[i] == '-' and zone_str[i - 1] not in ('e', 'E', '+', '-'):
            try:
                lo = float(zone_str[:i])
                hi = float(zone_str[i + 1:])
                return lo, hi
            except ValueError:
                continue
    return 0.0, 0.0


def run_backtest(
    client: BinanceFuturesClient,
    symbol: str,
    side_mode: str,
    interval: str = "4h",
    lookback_bars: int = 500,
    signal_lookback: int = 100,
    signal_cooldown: int = 12,
    fill_window: int = 12,
    trade_timeout: int = 48,
    atr_period: int = 14,
    adx_period: int = 14,
    range_lookback: int = 48,
    late_ok_atr: float = 0.5,
    late_watch_atr: float = 1.5,
) -> BacktestResult:
    """Run a walk-forward backtest of the WhaleTraderBot algo on historical klines.

    How it works:
    1. Download historical klines for the symbol
    2. Detect BTC regime from BTC 1D data (used as context throughout)
    3. Walk forward bar-by-bar starting from bar `signal_lookback`
    4. At each bar, run build_algo_plan() using the last `signal_lookback` bars
    5. If a signal is generated and no trade is active, look for entry fill
       in the next `fill_window` bars (price must touch the entry zone)
    6. Once filled, monitor for SL hit, TP hit, or timeout
    7. Collect all trades and compute statistics

    Args:
        client: BinanceFuturesClient instance
        symbol: e.g. "BTCUSDT"
        side_mode: "LONG" or "SHORT"
        interval: kline interval (default "4h")
        lookback_bars: total bars to download (default 500)
        signal_lookback: bars of history needed before generating first signal (default 100)
        signal_cooldown: minimum bars between signals (default 12 = ~2 days on 4h)
        fill_window: bars to wait for price to reach entry zone (default 12)
        trade_timeout: max bars to hold a trade before forced exit (default 48)
        atr_period: ATR period for algo
        adx_period: ADX period for algo
        range_lookback: range lookback for algo
        late_ok_atr: late OK threshold in ATR units
        late_watch_atr: late WATCH threshold in ATR units
    """
    console.print(f"[bold]Backtest: {symbol} {side_mode} ({interval})[/bold]")
    console.print(f"Downloading {lookback_bars} bars of {interval} klines...")

    # Download klines for the target symbol
    klines_raw = client.klines(symbol, interval, limit=lookback_bars)
    if len(klines_raw) < signal_lookback + 20:
        console.print(f"[red]Not enough data: got {len(klines_raw)} bars, need at least {signal_lookback + 20}[/red]")
        return _empty_result(symbol, side_mode, interval)

    open_arr = np.array([float(x[1]) for x in klines_raw], dtype=float)
    high_arr = np.array([float(x[2]) for x in klines_raw], dtype=float)
    low_arr = np.array([float(x[3]) for x in klines_raw], dtype=float)
    close_arr = np.array([float(x[4]) for x in klines_raw], dtype=float)
    volume_arr = np.array([float(x[5]) for x in klines_raw], dtype=float)
    times = [int(x[0]) for x in klines_raw]

    total_bars = len(close_arr)
    console.print(f"Loaded {total_bars} bars. Price range: {close_arr.min():.4f} - {close_arr.max():.4f}")

    # BTC regime (use 1D data for context — same as live pipeline)
    console.print("Fetching BTC 1D regime context...")
    btc_kl = client.klines("BTCUSDT", interval="1d", limit=250)
    btc_high = np.array([float(x[2]) for x in btc_kl], dtype=float)
    btc_low = np.array([float(x[3]) for x in btc_kl], dtype=float)
    btc_close = np.array([float(x[4]) for x in btc_kl], dtype=float)
    btc_open = np.array([float(x[1]) for x in btc_kl], dtype=float)
    btc_regime = detect_btc_regime(btc_open, btc_high, btc_low, btc_close)
    console.print(f"BTC regime: {btc_regime.btc_trend}")

    # Walk-forward simulation
    trades: List[Trade] = []
    signals_generated = 0
    last_signal_bar = -signal_cooldown  # allow first signal immediately
    active_trade: Optional[Dict[str, Any]] = None

    console.print(f"Running walk-forward from bar {signal_lookback} to {total_bars - 1}...")

    for bar_i in range(signal_lookback, total_bars):
        # --- Check active trade for exit conditions ---
        if active_trade is not None:
            at = active_trade
            bar_high = high_arr[bar_i]
            bar_low = low_arr[bar_i]
            bar_close = close_arr[bar_i]

            # Check if we're still waiting for fill
            if at["fill_index"] == -1:
                # Check if price enters the entry zone on this bar
                entry_lo = at["entry_low"]
                entry_hi = at["entry_high"]
                touched = False
                if side_mode == "LONG":
                    # For LONG: price must dip into entry zone (low <= entry_high)
                    touched = bar_low <= entry_hi
                else:
                    # For SHORT: price must rise into entry zone (high >= entry_low)
                    touched = bar_high >= entry_lo

                if touched:
                    # Fill at midpoint of entry zone
                    fill_price = (entry_lo + entry_hi) / 2.0
                    at["fill_index"] = bar_i
                    at["fill_price"] = fill_price
                elif bar_i - at["bar_index"] >= fill_window:
                    # Fill window expired — cancel this signal
                    active_trade = None
                continue

            # Trade is filled — check SL/TP
            fill_px = at["fill_price"]
            sl = at["stop_loss"]
            tps = at["take_profits"]
            bars_in_trade = bar_i - at["fill_index"]

            exit_price = None
            exit_reason = None
            tp_hit = 0

            if side_mode == "LONG":
                # Check SL first (worst case)
                if bar_low <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                # Check TPs (best to worst: TP3 > TP2 > TP1)
                elif len(tps) >= 3 and bar_high >= tps[2]:
                    exit_price = tps[2]
                    exit_reason = "TP3"
                    tp_hit = 3
                elif len(tps) >= 2 and bar_high >= tps[1]:
                    exit_price = tps[1]
                    exit_reason = "TP2"
                    tp_hit = 2
                elif len(tps) >= 1 and bar_high >= tps[0]:
                    exit_price = tps[0]
                    exit_reason = "TP1"
                    tp_hit = 1
                elif bars_in_trade >= trade_timeout:
                    exit_price = bar_close
                    exit_reason = "TIMEOUT"
            else:  # SHORT
                if bar_high >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif len(tps) >= 3 and bar_low <= tps[2]:
                    exit_price = tps[2]
                    exit_reason = "TP3"
                    tp_hit = 3
                elif len(tps) >= 2 and bar_low <= tps[1]:
                    exit_price = tps[1]
                    exit_reason = "TP2"
                    tp_hit = 2
                elif len(tps) >= 1 and bar_low <= tps[0]:
                    exit_price = tps[0]
                    exit_reason = "TP1"
                    tp_hit = 1
                elif bars_in_trade >= trade_timeout:
                    exit_price = bar_close
                    exit_reason = "TIMEOUT"

            if exit_price is not None:
                # Calculate PnL
                if side_mode == "LONG":
                    pnl_pct = (exit_price - fill_px) / fill_px * 100.0
                else:
                    pnl_pct = (fill_px - exit_price) / fill_px * 100.0

                risk = abs(fill_px - sl)
                r_mult = (exit_price - fill_px) / risk if risk > 0 and side_mode == "LONG" else (fill_px - exit_price) / risk if risk > 0 else 0.0

                trade = Trade(
                    bar_index=at["bar_index"],
                    fill_index=at["fill_index"],
                    exit_index=bar_i,
                    symbol=symbol,
                    side=side_mode,
                    setup_type=at["setup_type"],
                    entry_low=at["entry_low"],
                    entry_high=at["entry_high"],
                    fill_price=fill_px,
                    stop_loss=sl,
                    take_profits=tps,
                    tp_hit=tp_hit,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    r_multiple=r_mult,
                    algo_score=at["algo_score"],
                )
                trades.append(trade)
                active_trade = None

        # --- Generate new signal if no active trade and cooldown elapsed ---
        if active_trade is None and (bar_i - last_signal_bar) >= signal_cooldown:
            # Slice history up to current bar (inclusive)
            h_slice = high_arr[:bar_i + 1]
            l_slice = low_arr[:bar_i + 1]
            c_slice = close_arr[:bar_i + 1]
            cur_price = float(c_slice[-1])

            # Use recent volume as proxy for 24h volume
            vol_slice = volume_arr[max(0, bar_i - 5):bar_i + 1]
            vol_proxy = float(np.sum(vol_slice)) * cur_price  # rough USDT estimate

            plan = build_algo_plan(
                symbol=symbol,
                side_mode=side_mode,
                close=c_slice,
                high=h_slice,
                low=l_slice,
                current_price=cur_price,
                bid=cur_price * 0.9999,
                ask=cur_price * 1.0001,
                volume_usdt=vol_proxy,
                btc_trend=btc_regime.btc_trend,
                range_lookback=range_lookback,
                atr_period=atr_period,
                adx_period=adx_period,
                late_ok_atr=late_ok_atr,
                late_watch_atr=late_watch_atr,
            )

            signals_generated += 1
            last_signal_bar = bar_i

            # Only take signals where status is OK (not too late)
            if plan.status_hint == "OK" and plan.score_setup >= 40:
                entry_lo, entry_hi = _parse_entry_zone(plan.entry_zone)
                if entry_lo > 0 and entry_hi > 0:
                    active_trade = {
                        "bar_index": bar_i,
                        "fill_index": -1,
                        "fill_price": 0.0,
                        "entry_low": entry_lo,
                        "entry_high": entry_hi,
                        "stop_loss": plan.stop_loss,
                        "take_profits": plan.take_profits,
                        "setup_type": plan.setup_type,
                        "algo_score": plan.score_setup,
                    }

    # Close any remaining open trade at last bar
    if active_trade is not None and active_trade["fill_index"] != -1:
        at = active_trade
        fill_px = at["fill_price"]
        exit_price = float(close_arr[-1])
        sl = at["stop_loss"]
        if side_mode == "LONG":
            pnl_pct = (exit_price - fill_px) / fill_px * 100.0
        else:
            pnl_pct = (fill_px - exit_price) / fill_px * 100.0
        risk = abs(fill_px - sl)
        r_mult = (exit_price - fill_px) / risk if risk > 0 and side_mode == "LONG" else (fill_px - exit_price) / risk if risk > 0 else 0.0
        trade = Trade(
            bar_index=at["bar_index"],
            fill_index=at["fill_index"],
            exit_index=total_bars - 1,
            symbol=symbol,
            side=side_mode,
            setup_type=at["setup_type"],
            entry_low=at["entry_low"],
            entry_high=at["entry_high"],
            fill_price=fill_px,
            stop_loss=sl,
            take_profits=at["take_profits"],
            tp_hit=0,
            exit_price=exit_price,
            exit_reason="OPEN",
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            algo_score=at["algo_score"],
        )
        trades.append(trade)

    # Compute stats
    result = _compute_stats(symbol, side_mode, interval, total_bars, signals_generated, trades)
    _print_results(result)
    return result


def _empty_result(symbol: str, side: str, interval: str) -> BacktestResult:
    return BacktestResult(
        symbol=symbol, side=side, interval=interval, total_bars=0,
        signals_generated=0, trades_filled=0, trades_won=0, trades_lost=0,
        trades_breakeven=0, trades_timeout=0, win_rate=0.0, avg_r=0.0,
        total_r=0.0, max_r=0.0, min_r=0.0, profit_factor=0.0,
        avg_pnl_pct=0.0, total_pnl_pct=0.0, max_drawdown_r=0.0,
        trades=[], equity_curve=[],
    )


def _compute_stats(
    symbol: str, side: str, interval: str,
    total_bars: int, signals_generated: int,
    trades: List[Trade],
) -> BacktestResult:
    if not trades:
        result = _empty_result(symbol, side, interval)
        result.total_bars = total_bars
        result.signals_generated = signals_generated
        return result

    filled = [t for t in trades if t.fill_price > 0]
    won = [t for t in filled if t.r_multiple > 0.05]
    lost = [t for t in filled if t.r_multiple < -0.05]
    be = [t for t in filled if -0.05 <= t.r_multiple <= 0.05]
    timeouts = [t for t in filled if t.exit_reason == "TIMEOUT"]

    r_values = [t.r_multiple for t in filled]
    pnl_values = [t.pnl_pct for t in filled]

    total_r = sum(r_values)
    avg_r = np.mean(r_values) if r_values else 0.0
    max_r = max(r_values) if r_values else 0.0
    min_r = min(r_values) if r_values else 0.0

    gross_profit = sum(r for r in r_values if r > 0)
    gross_loss = abs(sum(r for r in r_values if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Equity curve & drawdown (in R units)
    equity = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        equity.append(cumulative)
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    win_rate = len(won) / len(filled) if filled else 0.0

    return BacktestResult(
        symbol=symbol,
        side=side,
        interval=interval,
        total_bars=total_bars,
        signals_generated=signals_generated,
        trades_filled=len(filled),
        trades_won=len(won),
        trades_lost=len(lost),
        trades_breakeven=len(be),
        trades_timeout=len(timeouts),
        win_rate=win_rate,
        avg_r=float(avg_r),
        total_r=float(total_r),
        max_r=float(max_r),
        min_r=float(min_r),
        profit_factor=float(profit_factor),
        avg_pnl_pct=float(np.mean(pnl_values)) if pnl_values else 0.0,
        total_pnl_pct=float(sum(pnl_values)),
        max_drawdown_r=float(max_dd),
        trades=trades,
        equity_curve=equity,
    )


def _print_results(result: BacktestResult) -> None:
    console.print()
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]  BACKTEST RESULTS: {result.symbol} {result.side} ({result.interval})[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Bars", str(result.total_bars))
    table.add_row("Signals Generated", str(result.signals_generated))
    table.add_row("Trades Filled", str(result.trades_filled))
    table.add_row("Trades Won", f"[green]{result.trades_won}[/green]")
    table.add_row("Trades Lost", f"[red]{result.trades_lost}[/red]")
    table.add_row("Trades Breakeven", str(result.trades_breakeven))
    table.add_row("Trades Timeout", str(result.trades_timeout))
    table.add_row("", "")
    table.add_row("Win Rate", f"[bold]{result.win_rate:.1%}[/bold]")
    table.add_row("Average R", f"[bold]{result.avg_r:+.3f}R[/bold]")
    table.add_row("Total R", f"[bold {'green' if result.total_r > 0 else 'red'}]{result.total_r:+.3f}R[/bold {'green' if result.total_r > 0 else 'red'}]")
    table.add_row("Best Trade", f"[green]{result.max_r:+.3f}R[/green]")
    table.add_row("Worst Trade", f"[red]{result.min_r:+.3f}R[/red]")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("", "")
    table.add_row("Avg PnL %", f"{result.avg_pnl_pct:+.3f}%")
    table.add_row("Total PnL %", f"[bold]{result.total_pnl_pct:+.3f}%[/bold]")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown_r:.3f}R[/red]")

    console.print(table)
    console.print()

    # Trade log
    if result.trades:
        trade_table = Table(title="Trade Log")
        trade_table.add_column("#", justify="right")
        trade_table.add_column("Setup")
        trade_table.add_column("Fill Price", justify="right")
        trade_table.add_column("Exit Price", justify="right")
        trade_table.add_column("Exit Reason")
        trade_table.add_column("PnL %", justify="right")
        trade_table.add_column("R-Multiple", justify="right")
        trade_table.add_column("Bars Held", justify="right")

        for i, t in enumerate(result.trades, 1):
            bars_held = t.exit_index - t.fill_index if t.fill_index >= 0 else 0
            pnl_style = "green" if t.pnl_pct > 0 else "red" if t.pnl_pct < 0 else "white"
            reason_style = "green" if "TP" in t.exit_reason else "red" if t.exit_reason == "SL" else "yellow"

            trade_table.add_row(
                str(i),
                t.setup_type,
                f"{t.fill_price:.4f}",
                f"{t.exit_price:.4f}",
                f"[{reason_style}]{t.exit_reason}[/{reason_style}]",
                f"[{pnl_style}]{t.pnl_pct:+.3f}%[/{pnl_style}]",
                f"[{pnl_style}]{t.r_multiple:+.3f}R[/{pnl_style}]",
                str(bars_held),
            )

        console.print(trade_table)
        console.print()

    # Setup type breakdown
    if result.trades:
        setup_stats: Dict[str, List[float]] = {}
        for t in result.trades:
            setup_stats.setdefault(t.setup_type, []).append(t.r_multiple)

        breakdown = Table(title="Performance by Setup Type")
        breakdown.add_column("Setup Type")
        breakdown.add_column("Count", justify="right")
        breakdown.add_column("Win Rate", justify="right")
        breakdown.add_column("Avg R", justify="right")
        breakdown.add_column("Total R", justify="right")

        for st, rs in sorted(setup_stats.items()):
            wins = sum(1 for r in rs if r > 0.05)
            wr = wins / len(rs) if rs else 0
            breakdown.add_row(
                st,
                str(len(rs)),
                f"{wr:.1%}",
                f"{np.mean(rs):+.3f}R",
                f"{sum(rs):+.3f}R",
            )
        console.print(breakdown)


def run_multi_backtest(
    client: BinanceFuturesClient,
    symbols: List[str],
    side_mode: str,
    interval: str = "4h",
    **kwargs: Any,
) -> List[BacktestResult]:
    """Run backtest across multiple symbols and print a summary."""
    results: List[BacktestResult] = []
    for sym in symbols:
        try:
            result = run_backtest(client, sym, side_mode, interval, **kwargs)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error backtesting {sym}: {e}[/red]")

    if len(results) > 1:
        _print_portfolio_summary(results)

    return results


def _print_portfolio_summary(results: List[BacktestResult]) -> None:
    console.print()
    console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")
    console.print(f"[bold magenta]  PORTFOLIO BACKTEST SUMMARY[/bold magenta]")
    console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")

    summary = Table(title="Summary by Symbol")
    summary.add_column("Symbol")
    summary.add_column("Trades", justify="right")
    summary.add_column("Win Rate", justify="right")
    summary.add_column("Total R", justify="right")
    summary.add_column("Avg R", justify="right")
    summary.add_column("PF", justify="right")
    summary.add_column("Max DD (R)", justify="right")

    total_trades = 0
    total_won = 0
    total_r = 0.0
    all_r_values: List[float] = []

    for r in results:
        style = "green" if r.total_r > 0 else "red"
        summary.add_row(
            r.symbol,
            str(r.trades_filled),
            f"{r.win_rate:.1%}",
            f"[{style}]{r.total_r:+.2f}R[/{style}]",
            f"{r.avg_r:+.3f}R",
            f"{r.profit_factor:.2f}",
            f"{r.max_drawdown_r:.2f}R",
        )
        total_trades += r.trades_filled
        total_won += r.trades_won
        total_r += r.total_r
        all_r_values.extend(t.r_multiple for t in r.trades)

    console.print(summary)

    if total_trades > 0:
        console.print()
        console.print(f"[bold]Portfolio Totals:[/bold]")
        console.print(f"  Total Trades: {total_trades}")
        console.print(f"  Overall Win Rate: {total_won / total_trades:.1%}")
        r_style = "green" if total_r > 0 else "red"
        console.print(f"  Total R: [{r_style}]{total_r:+.2f}R[/{r_style}]")
        console.print(f"  Avg R per Trade: {np.mean(all_r_values):+.3f}R")

        gross_p = sum(r for r in all_r_values if r > 0)
        gross_l = abs(sum(r for r in all_r_values if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float('inf') if gross_p > 0 else 0.0
        console.print(f"  Portfolio Profit Factor: {pf:.2f}")
    console.print()
