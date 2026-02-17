"""Level Engine — deterministic pivot clusters, AVWAP anchors, magnets, and
entry/SL/TP refinement per tactic.

All functions accept numpy arrays and return deterministic results.
No lookahead: only CLOSED bars are used (arrays must be pre-sliced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .indicators import (
    EPS,
    anchored_vwap,
    atr,
    bollinger_bands,
    bollinger_bandwidth,
    choppiness_index,
    ema,
    typical_price,
    true_range,
    vwap,
)
from .structure import Pivots, detect_pivots

# ---------------------------------------------------------------------------
# 2.1  Pivot clustering
# ---------------------------------------------------------------------------


@dataclass
class PivotCluster:
    level_price: float
    density: int
    kind: str  # "support", "resistance", "mixed"
    newest_bar_idx: int  # for recency sorting


def cluster_pivots(
    pivots: Pivots,
    atr_last: float,
    tol_atr: float = 0.30,
    max_clusters: int = 12,
) -> List[PivotCluster]:
    """Cluster nearby pivot prices using ATR-based tolerance.

    Returns clusters sorted by density desc then recency desc, capped at
    *max_clusters*.
    """
    tol = tol_atr * max(atr_last, EPS)

    # Collect all pivot points with their kind
    points: List[Tuple[float, str, int]] = []
    for idx, price in pivots.highs:
        points.append((price, "resistance", idx))
    for idx, price in pivots.lows:
        points.append((price, "support", idx))

    if not points:
        return []

    # Sort by price for clustering
    points.sort(key=lambda x: x[0])

    clusters: List[PivotCluster] = []
    used = [False] * len(points)

    for i in range(len(points)):
        if used[i]:
            continue
        group_prices = [points[i][0]]
        group_kinds = {points[i][1]}
        group_newest = points[i][2]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            if abs(points[j][0] - np.median(group_prices)) <= tol:
                group_prices.append(points[j][0])
                group_kinds.add(points[j][1])
                group_newest = max(group_newest, points[j][2])
                used[j] = True

        kind = "mixed" if len(group_kinds) > 1 else next(iter(group_kinds))
        clusters.append(
            PivotCluster(
                level_price=float(np.median(group_prices)),
                density=len(group_prices),
                kind=kind,
                newest_bar_idx=group_newest,
            )
        )

    # Sort: density desc, then recency desc
    clusters.sort(key=lambda c: (-c.density, -c.newest_bar_idx))
    return clusters[:max_clusters]


# ---------------------------------------------------------------------------
# 2.2  Deterministic AVWAP anchor selection per tactic
# ---------------------------------------------------------------------------


def _find_last_swing_low(pivots: Pivots) -> Optional[int]:
    """Index of most recent pivot low."""
    if not pivots.lows:
        return None
    return pivots.lows[-1][0]


def _find_last_swing_high(pivots: Pivots) -> Optional[int]:
    """Index of most recent pivot high."""
    if not pivots.highs:
        return None
    return pivots.highs[-1][0]


def _find_sweep_candle(
    side: str,
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    rng_low: float,
    rng_high: float,
    atr_last: float,
    lookback: int = 20,
) -> Optional[int]:
    """Find the bar index where price swept beyond range edge then reclaimed."""
    n = len(close)
    start = max(0, n - lookback)
    sweep_buf = 0.15 * atr_last
    if side == "LONG":
        for i in range(n - 1, start - 1, -1):
            if float(low[i]) < rng_low - sweep_buf and float(close[i]) > rng_low:
                return i
    else:
        for i in range(n - 1, start - 1, -1):
            if float(high[i]) > rng_high + sweep_buf and float(close[i]) < rng_high:
                return i
    return None


def _find_breakout_candle(
    side: str,
    close: np.ndarray,
    rng_low: float,
    rng_high: float,
    atr_last: float,
    lookback: int = 20,
) -> Optional[int]:
    """Find the first bar (most recent N) that closed outside the range."""
    n = len(close)
    start = max(0, n - lookback)
    buf = 0.10 * atr_last
    if side == "LONG":
        for i in range(start, n):
            if float(close[i]) > rng_high + buf:
                return i
    else:
        for i in range(start, n):
            if float(close[i]) < rng_low - buf:
                return i
    return None


def _find_largest_tr_candle(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 30
) -> int:
    """Index of the candle with the largest true range in last *lookback* bars."""
    n = len(close)
    start = max(0, n - lookback)
    tr_arr = true_range(high, low, close)
    subset = tr_arr[start:]
    if len(subset) == 0:
        return max(0, n - 1)
    return int(start + np.argmax(subset))


def choose_avwap_anchor(
    tactic: str,
    side: str,
    pivots: Pivots,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    rng_low: float,
    rng_high: float,
    atr_last: float,
    fallback_window: int = 60,
) -> Tuple[int, str]:
    """Return (anchor_bar_index, anchor_type) for AVWAP per tactic.

    Falls back to ``max(0, len(close) - fallback_window)`` if the
    event cannot be detected.
    """
    n = len(close)
    default = (max(0, n - fallback_window), "FALLBACK")

    if tactic == "TREND_PULLBACK":
        if side == "LONG":
            idx = _find_last_swing_low(pivots)
        else:
            idx = _find_last_swing_high(pivots)
        if idx is not None:
            return (idx, "IMPULSE_START")
        return default

    elif tactic == "RANGE_SWEEP_RECLAIM":
        idx = _find_sweep_candle(side, low, high, close, rng_low, rng_high, atr_last)
        if idx is not None:
            return (idx, "SWEEP_CANDLE")
        return default

    elif tactic == "BREAKOUT_RETEST":
        idx = _find_breakout_candle(side, close, rng_low, rng_high, atr_last)
        if idx is not None:
            return (idx, "BREAKOUT_CANDLE")
        return default

    elif tactic == "VOLATILITY_FADE":
        idx = _find_largest_tr_candle(high, low, close, lookback=30)
        return (idx, "LARGEST_TR")

    return default


# ---------------------------------------------------------------------------
# 2.3  Liquidity magnets builder
# ---------------------------------------------------------------------------


@dataclass
class Magnet:
    price: float
    tag: str
    priority: float  # higher = more significant


def build_magnets(
    rng_low: float,
    rng_high: float,
    clusters: List[PivotCluster],
    avwap_last: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    top_k_clusters: int = 5,
) -> List[Magnet]:
    """Assemble list of price magnets from multiple sources."""
    magnets: List[Magnet] = []
    rng_mid = (rng_low + rng_high) / 2.0

    magnets.append(Magnet(price=rng_low, tag="RANGE_LOW", priority=3.0))
    magnets.append(Magnet(price=rng_high, tag="RANGE_HIGH", priority=3.0))
    magnets.append(Magnet(price=rng_mid, tag="RANGE_MID", priority=2.0))

    for i, c in enumerate(clusters[:top_k_clusters]):
        magnets.append(
            Magnet(
                price=c.level_price,
                tag=f"PIVOT_{c.kind.upper()}",
                priority=1.5 + min(c.density, 5) * 0.3,
            )
        )

    if avwap_last is not None and np.isfinite(avwap_last):
        magnets.append(Magnet(price=avwap_last, tag="AVWAP", priority=2.5))

    if bb_upper is not None and np.isfinite(bb_upper):
        magnets.append(Magnet(price=bb_upper, tag="BB_UPPER", priority=1.5))
    if bb_lower is not None and np.isfinite(bb_lower):
        magnets.append(Magnet(price=bb_lower, tag="BB_LOWER", priority=1.5))

    return magnets


def _pick_tp_magnets(
    magnets: List[Magnet],
    entry_mid: float,
    side: str,
    min_r: float,
    risk: float,
    count: int = 3,
) -> List[Magnet]:
    """Select magnets in the profit direction, sorted by distance from entry.

    Only includes magnets that are at least *min_r * risk* away from *entry_mid*.
    """
    if risk <= 0:
        return []

    candidates: List[Magnet] = []
    for m in magnets:
        dist = m.price - entry_mid if side == "LONG" else entry_mid - m.price
        if dist >= min_r * risk:
            candidates.append(m)

    # Sort by distance from entry (ascending)
    candidates.sort(
        key=lambda m: abs(m.price - entry_mid)
    )
    return candidates[:count]


# ---------------------------------------------------------------------------
# 2.4  Refine entry_zone, SL, TP per tactic
# ---------------------------------------------------------------------------


@dataclass
class LevelsDebug:
    tactic: str
    avwap_anchor_type: str
    avwap_anchor_idx: int
    avwap_status: str  # "OK" or "NO_VOLUME_FALLBACK"
    chop: Optional[float]
    bb_width: Optional[float]
    magnets_used: List[Dict[str, Any]]
    repaired: bool = False
    repair_reason: str = ""


@dataclass
class RefinedLevels:
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profits: List[float]  # [TP1, TP2, TP3]
    debug: LevelsDebug


def refine_plan_for_tactic(
    tactic: str,
    side: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray],
    atr_last: float,
    atr_period: int,
    rng_low: float,
    rng_high: float,
    rng_h: float,
    existing_entry_low: float,
    existing_entry_high: float,
    existing_sl: float,
    existing_tps: List[float],
    pivot_w: int = 2,
    bb_period: int = 20,
    bb_std: float = 2.0,
    chop_period: int = 14,
    adx_last: float = 0.0,
    entry_width_cap_atr: float = 1.5,
    min_r_tp1: float = 0.8,
    min_r_tp2: float = 1.5,
    min_r_tp3: float = 2.2,
) -> RefinedLevels:
    """Refine entry/SL/TP for a given tactic using the Level Engine.

    Combines pivot clusters, AVWAP, Bollinger Bands, and structure magnets
    to improve accuracy over the raw algo plan while preserving fallback
    to existing levels when data is insufficient.
    """
    n = len(close)

    # --- Pivots & clusters ---
    pivots = detect_pivots(high, low, w=pivot_w) if n >= 2 * pivot_w + 1 else Pivots([], [])
    clusters = cluster_pivots(pivots, atr_last)

    # --- AVWAP ---
    has_volume = volume is not None and len(volume) == n and np.any(volume > 0)
    avwap_status = "OK" if has_volume else "NO_VOLUME_FALLBACK"

    anchor_idx, anchor_type = choose_avwap_anchor(
        tactic, side, pivots, high, low, close, rng_low, rng_high, atr_last
    )

    avwap_last: Optional[float] = None
    if has_volume:
        tp = typical_price(high, low, close)
        avwap_series = anchored_vwap(tp, volume, anchor_idx)
        val = avwap_series[-1]
        if np.isfinite(val):
            avwap_last = float(val)

    # --- Bollinger Bands ---
    bb_upper_arr, bb_mid_arr, bb_lower_arr = bollinger_bands(close, bb_period, bb_std)
    bb_upper_last = float(bb_upper_arr[-1]) if np.isfinite(bb_upper_arr[-1]) else None
    bb_lower_last = float(bb_lower_arr[-1]) if np.isfinite(bb_lower_arr[-1]) else None
    bb_mid_last = float(bb_mid_arr[-1]) if np.isfinite(bb_mid_arr[-1]) else None

    # BBW and CHOP for debug output
    bbw_arr = bollinger_bandwidth(bb_upper_arr, bb_mid_arr, bb_lower_arr)
    bbw_last = float(bbw_arr[-1]) if np.isfinite(bbw_arr[-1]) else None

    chop_arr = choppiness_index(high, low, close, chop_period)
    chop_last = float(chop_arr[-1]) if np.isfinite(chop_arr[-1]) else None

    # --- Magnets ---
    magnets = build_magnets(
        rng_low, rng_high, clusters, avwap_last, bb_upper_last, bb_lower_last
    )

    # --- Entry refinement ---
    # Blend existing entry with AVWAP and nearest pivot cluster
    entry_low = existing_entry_low
    entry_high = existing_entry_high

    # AVWAP blending: nudge entry toward AVWAP when it's within 1 ATR
    if avwap_last is not None:
        avwap_dist = abs(avwap_last - (entry_low + entry_high) / 2.0)
        if avwap_dist <= 1.0 * atr_last:
            blend_weight = 0.25  # conservative blend
            entry_mid_old = (entry_low + entry_high) / 2.0
            entry_mid_new = entry_mid_old * (1 - blend_weight) + avwap_last * blend_weight
            half_w = (entry_high - entry_low) / 2.0
            entry_low = entry_mid_new - half_w
            entry_high = entry_mid_new + half_w

    # Nearest pivot cluster blending
    entry_mid = (entry_low + entry_high) / 2.0
    nearest_cluster = _nearest_cluster_to_price(clusters, entry_mid, side, atr_last)
    if nearest_cluster is not None:
        cluster_dist = abs(nearest_cluster.level_price - entry_mid)
        if cluster_dist <= 0.8 * atr_last:
            # Nudge toward cluster level (higher density = stronger pull)
            pull = min(0.20, 0.05 * nearest_cluster.density)
            entry_mid_new = entry_mid * (1 - pull) + nearest_cluster.level_price * pull
            half_w = (entry_high - entry_low) / 2.0
            entry_low = entry_mid_new - half_w
            entry_high = entry_mid_new + half_w

    # Cap entry zone width to entry_width_cap_atr * ATR
    max_width = entry_width_cap_atr * atr_last
    current_width = entry_high - entry_low
    if current_width > max_width and current_width > 0:
        entry_mid = (entry_low + entry_high) / 2.0
        entry_low = entry_mid - max_width / 2.0
        entry_high = entry_mid + max_width / 2.0

    # --- SL refinement ---
    # Place SL at thesis invalidation + noise buffer
    entry_mid = (entry_low + entry_high) / 2.0
    noise_buffer = max(0.15 * atr_last, 0.001)

    sl = existing_sl  # start with existing

    # Tactic-specific SL placement
    if tactic == "TREND_PULLBACK":
        # Beyond the nearest support/resistance cluster that invalidates thesis
        invalidation = _find_invalidation_level(clusters, entry_mid, side, atr_last)
        if invalidation is not None:
            if side == "LONG":
                sl = min(sl, invalidation - noise_buffer)
            else:
                sl = max(sl, invalidation + noise_buffer)

    elif tactic == "RANGE_SWEEP_RECLAIM":
        # SL beyond swept level + buffer
        if side == "LONG":
            sl = min(sl, rng_low - 0.35 * atr_last - noise_buffer)
        else:
            sl = max(sl, rng_high + 0.35 * atr_last + noise_buffer)

    elif tactic == "BREAKOUT_RETEST":
        # SL back inside range = thesis invalidated
        if side == "LONG":
            sl_candidate = rng_high - 0.5 * atr_last - noise_buffer
            sl = min(sl, sl_candidate)
        else:
            sl_candidate = rng_low + 0.5 * atr_last + noise_buffer
            sl = max(sl, sl_candidate)

    elif tactic == "VOLATILITY_FADE":
        # Keep existing SL (tight, near range edge) but add noise buffer
        if side == "LONG":
            sl = min(sl, entry_low - 0.4 * atr_last)
        else:
            sl = max(sl, entry_high + 0.4 * atr_last)

    # --- TP refinement via magnets ---
    risk = max(abs(entry_mid - sl), EPS)

    # Is market trending? (low CHOP + higher ADX = allow runner TP3)
    trending_strong = (
        (chop_last is not None and chop_last < 45.0) or adx_last >= 30
    )

    tp_magnets = _pick_tp_magnets(magnets, entry_mid, side, min_r_tp1, risk)

    tp1 = existing_tps[0] if len(existing_tps) >= 1 else entry_mid
    tp2 = existing_tps[1] if len(existing_tps) >= 2 else entry_mid
    tp3 = existing_tps[2] if len(existing_tps) >= 3 else entry_mid

    # Use magnet-based TPs when available, otherwise keep existing
    if len(tp_magnets) >= 1:
        tp1 = tp_magnets[0].price
    if len(tp_magnets) >= 2:
        tp2 = tp_magnets[1].price
    if len(tp_magnets) >= 3 and trending_strong:
        tp3 = tp_magnets[2].price

    # Ensure minimum R thresholds
    tp1 = _enforce_min_r(tp1, entry_mid, sl, side, min_r_tp1)
    tp2 = _enforce_min_r(tp2, entry_mid, sl, side, min_r_tp2)
    tp3 = _enforce_min_r(tp3, entry_mid, sl, side, min_r_tp3)

    # If not strongly trending, cap TP3 at more conservative level
    if not trending_strong:
        max_r_tp3 = 3.5
        if side == "LONG":
            tp3 = min(tp3, entry_mid + max_r_tp3 * risk)
        else:
            tp3 = max(tp3, entry_mid - max_r_tp3 * risk)

    tps = [tp1, tp2, tp3]

    # Build magnets debug info
    magnets_debug = [{"price": m.price, "tag": m.tag} for m in tp_magnets[:6]]

    debug = LevelsDebug(
        tactic=tactic,
        avwap_anchor_type=anchor_type,
        avwap_anchor_idx=anchor_idx,
        avwap_status=avwap_status,
        chop=chop_last,
        bb_width=bbw_last,
        magnets_used=magnets_debug,
    )

    return RefinedLevels(
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=sl,
        take_profits=tps,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_cluster_to_price(
    clusters: List[PivotCluster],
    price: float,
    side: str,
    atr_last: float,
) -> Optional[PivotCluster]:
    """Find the nearest pivot cluster to *price* on the entry side."""
    best = None
    best_dist = float("inf")
    for c in clusters:
        # For LONG, prefer clusters at or below price (support)
        # For SHORT, prefer clusters at or above price (resistance)
        if side == "LONG" and c.level_price > price + 0.5 * atr_last:
            continue
        if side == "SHORT" and c.level_price < price - 0.5 * atr_last:
            continue
        dist = abs(c.level_price - price)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def _find_invalidation_level(
    clusters: List[PivotCluster],
    entry_mid: float,
    side: str,
    atr_last: float,
) -> Optional[float]:
    """Find the nearest cluster beyond the entry that would invalidate the thesis."""
    candidates = []
    for c in clusters:
        if side == "LONG" and c.level_price < entry_mid:
            candidates.append(c.level_price)
        elif side == "SHORT" and c.level_price > entry_mid:
            candidates.append(c.level_price)

    if not candidates:
        return None

    if side == "LONG":
        # Nearest support below entry
        return max(candidates)
    else:
        # Nearest resistance above entry
        return min(candidates)


def _enforce_min_r(
    tp: float,
    entry_mid: float,
    sl: float,
    side: str,
    min_r: float,
) -> float:
    """Ensure TP is at least min_r * risk away from entry."""
    risk = max(abs(entry_mid - sl), EPS)
    min_dist = min_r * risk

    if side == "LONG":
        return max(tp, entry_mid + min_dist)
    else:
        return min(tp, entry_mid - min_dist)


# ---------------------------------------------------------------------------
# 2.5  Validation guards (centralized)
# ---------------------------------------------------------------------------


def validate_plan_levels(
    side: str,
    entry_low: float,
    entry_high: float,
    sl: float,
    tps: List[float],
    current_price: float,
) -> Tuple[float, float, float, List[float], bool, str]:
    """Validate and repair plan levels.

    Enforces:
    - LONG: SL < entry_low < entry_high < TP1 < TP2 < TP3
    - SHORT: TP3 < TP2 < TP1 < entry_low < entry_high < SL
    - All prices > 0
    - TP monotonicity

    Returns:
        (entry_low, entry_high, sl, tps, repaired, repair_reason)
    """
    repaired = False
    reasons: List[str] = []

    # Ensure entry ordering
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low
        repaired = True
        reasons.append("ENTRY_SWAPPED")

    entry_mid = (entry_low + entry_high) / 2.0
    risk = max(abs(entry_mid - sl), EPS)

    # Clamp all prices to > 0
    if entry_low <= 0:
        entry_low = max(entry_low, EPS)
        repaired = True
        reasons.append("ENTRY_LOW_CLAMPED")
    if entry_high <= 0:
        entry_high = max(entry_high, EPS)
        repaired = True
        reasons.append("ENTRY_HIGH_CLAMPED")
    if sl <= 0 and side == "LONG":
        sl = max(EPS, entry_low * 0.95)
        repaired = True
        reasons.append("SL_CLAMPED_POSITIVE")

    # Enforce SL side
    if side == "LONG":
        if sl >= entry_low:
            sl = entry_low - max(risk * 0.5, EPS)
            repaired = True
            reasons.append("SL_MOVED_BELOW_ENTRY")
    else:
        if sl <= entry_high:
            sl = entry_high + max(risk * 0.5, EPS)
            repaired = True
            reasons.append("SL_MOVED_ABOVE_ENTRY")

    # Re-compute risk after potential SL fix
    entry_mid = (entry_low + entry_high) / 2.0
    risk = max(abs(entry_mid - sl), EPS)

    # Validate TPs
    tps = list(tps)
    while len(tps) < 3:
        # Fill missing TPs with R-multiple fallback
        r_mult = [1.5, 2.5, 3.5][len(tps)]
        if side == "LONG":
            tps.append(entry_mid + r_mult * risk)
        else:
            tps.append(entry_mid - r_mult * risk)

    # Clamp negative TPs
    for i in range(len(tps)):
        if tps[i] <= 0:
            if side == "LONG":
                tps[i] = entry_mid + (1.0 + i) * risk
            else:
                tps[i] = max(EPS, entry_mid - (1.0 + i) * risk)
            repaired = True
            reasons.append(f"NEGATIVE_TP{i+1}")

    # Enforce TP ordering and side
    if side == "LONG":
        # All TPs must be above entry_high
        for i in range(len(tps)):
            if tps[i] <= entry_high:
                tps[i] = entry_high + (0.5 + i * 0.5) * risk
                repaired = True
                reasons.append(f"TP{i+1}_BELOW_ENTRY")
        # Monotonic: TP1 < TP2 < TP3
        for i in range(1, len(tps)):
            if tps[i] <= tps[i - 1]:
                tps[i] = tps[i - 1] + 0.3 * risk
                repaired = True
                reasons.append(f"TP{i+1}_MONOTONICITY")
    else:  # SHORT
        # All TPs must be below entry_low
        for i in range(len(tps)):
            if tps[i] >= entry_low:
                tps[i] = entry_low - (0.5 + i * 0.5) * risk
                repaired = True
                reasons.append(f"TP{i+1}_ABOVE_ENTRY")
        # Monotonic: TP1 > TP2 > TP3 (i.e., TP1 is closest to entry)
        for i in range(1, len(tps)):
            if tps[i] >= tps[i - 1]:
                tps[i] = tps[i - 1] - 0.3 * risk
                repaired = True
                reasons.append(f"TP{i+1}_MONOTONICITY")

    # Final negative TP guard (micro-priced assets)
    for i in range(len(tps)):
        if tps[i] <= 0:
            tps[i] = max(EPS, abs(tps[i]))
            repaired = True
            reasons.append(f"TP{i+1}_FINAL_CLAMP")

    repair_reason = "|".join(reasons) if reasons else ""
    return entry_low, entry_high, sl, tps, repaired, repair_reason
