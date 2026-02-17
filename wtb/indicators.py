from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1 - alpha) * out[i - 1]
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close, dtype=float)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    # Wilder smoothing
    out = np.zeros_like(close, dtype=float)
    out[0] = tr[0]
    for i in range(1, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Simplified ADX (Wilder). Returns ADX series."""
    n = len(close)
    if n < period + 2:
        return np.zeros_like(close, dtype=float)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

    # Wilder smoothing
    def wilder_smooth(x: np.ndarray, p: int) -> np.ndarray:
        out = np.zeros(n - 1, dtype=float)
        out[0] = x[0]
        for i in range(1, len(out)):
            out[i] = (out[i - 1] * (p - 1) + x[i]) / p
        return out

    tr_s = wilder_smooth(tr, period)
    plus_s = wilder_smooth(plus_dm, period)
    minus_s = wilder_smooth(minus_dm, period)

    # Avoid div by zero
    tr_s = np.where(tr_s == 0, 1e-12, tr_s)

    plus_di = 100 * (plus_s / tr_s)
    minus_di = 100 * (minus_s / tr_s)

    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)

    adx_s = np.zeros_like(dx)
    adx_s[0] = dx[0]
    for i in range(1, len(dx)):
        adx_s[i] = (adx_s[i - 1] * (period - 1) + dx[i]) / period

    # align length to close
    out = np.concatenate([[0.0], adx_s])
    return out


def returns(close: np.ndarray) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close, dtype=float)
    r = np.zeros_like(close, dtype=float)
    r[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    return r


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> float:
    if len(a) < window or len(b) < window:
        return 0.0
    x = a[-window:]
    y = b[-window:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Indicator primitives for Level Engine
# ---------------------------------------------------------------------------

EPS = 1e-12


def typical_price(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """(High + Low + Close) / 3."""
    return (high + low + close) / 3.0


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range series (same length as input, first bar = H-L)."""
    if len(close) < 2:
        return high - low
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return np.concatenate([[high[0] - low[0]], tr])


def bollinger_bands(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, middle, lower).

    Uses simple moving average. Returns arrays of same length as *close*;
    values before index ``period - 1`` are filled with NaN.

    Reference: TradingView / Fidelity BBW docs.
    """
    n = len(close)
    upper = np.full(n, np.nan, dtype=float)
    middle = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)
    if n < period:
        return upper, middle, lower
    # rolling mean + std via cumsum trick
    cs = np.cumsum(close)
    cs2 = np.cumsum(close ** 2)
    for i in range(period - 1, n):
        s = cs[i] - (cs[i - period] if i >= period else 0.0)
        s2 = cs2[i] - (cs2[i - period] if i >= period else 0.0)
        m = s / period
        var = s2 / period - m * m
        std = np.sqrt(max(var, 0.0))
        middle[i] = m
        upper[i] = m + num_std * std
        lower[i] = m - num_std * std
    return upper, middle, lower


def bollinger_bandwidth(
    upper: np.ndarray, middle: np.ndarray, lower: np.ndarray
) -> np.ndarray:
    """Bollinger BandWidth = (Upper - Lower) / Middle * 100.

    Per TradingView & Fidelity definition.
    """
    return (upper - lower) / np.maximum(np.abs(middle), EPS) * 100.0


def vwap(tp: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Session VWAP from typical price and volume (cumulative from bar 0)."""
    vol = np.maximum(volume, EPS)
    cum_pv = np.cumsum(tp * vol)
    cum_v = np.cumsum(vol)
    return cum_pv / np.maximum(cum_v, EPS)


def anchored_vwap(tp: np.ndarray, volume: np.ndarray, anchor_idx: int) -> np.ndarray:
    """Anchored VWAP starting from *anchor_idx*.

    Bars before the anchor are NaN. Reference: TradingView Anchored VWAP docs.
    """
    n = len(tp)
    anchor_idx = int(np.clip(anchor_idx, 0, max(0, n - 1)))
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    sub_tp = tp[anchor_idx:]
    sub_vol = np.maximum(volume[anchor_idx:], EPS)
    out[anchor_idx:] = np.cumsum(sub_tp * sub_vol) / np.maximum(np.cumsum(sub_vol), EPS)
    return out


def choppiness_index(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Choppiness Index (CHOP) per TradingView docs.

    CHOP = 100 * LOG10(SUM(TR, period) / (Highest_High - Lowest_Low)) / LOG10(period)

    High values (>61.8) → choppy/ranging market.
    Low values (<38.2) → trending market.

    Returns array same length as input; values before ``period`` are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out

    tr = true_range(high, low, close)
    log_period = np.log10(float(period))
    if log_period == 0:
        return out

    # rolling sum of TR, rolling highest high, rolling lowest low
    tr_cs = np.cumsum(tr)
    for i in range(period, n):
        tr_sum = tr_cs[i] - tr_cs[i - period]
        hh = np.max(high[i - period + 1 : i + 1])
        ll = np.min(low[i - period + 1 : i + 1])
        hl_range = max(hh - ll, EPS)
        out[i] = 100.0 * np.log10(tr_sum / hl_range) / log_period

    return out
