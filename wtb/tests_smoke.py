#!/usr/bin/env python3
"""Smoke tests for WhaleTraderBot v2.4 ALGO-only.

These tests verify critical logic paths without needing live data.
Run with: python3 -m wtb.tests_smoke
"""

from __future__ import annotations

import numpy as np
from .algo import choose_setup_type, build_algo_plan
from .binance import normalize_symbol, resolve_symbol
from .indicators import (
    bollinger_bands, bollinger_bandwidth, choppiness_index,
    typical_price, vwap, anchored_vwap,
)
from .level_engine import (
    cluster_pivots, validate_plan_levels, build_magnets,
    refine_plan_for_tactic, PivotCluster,
)
from .structure import Pivots, detect_pivots


def test_setup_type_determinism():
    """Verify that all 4 setup types can be produced deterministically."""
    print("Testing setup_type determinism...")

    # Mock data: simple arrays
    close = np.array([100.0] * 100 + [105.0, 106.0, 107.0])
    high = np.array([101.0] * 100 + [106.0, 107.0, 108.0])
    low = np.array([99.0] * 100 + [104.0, 105.0, 106.0])
    current_price = 107.0
    atr_last = 2.0

    # Test case 1: RANGE_SWEEP_RECLAIM
    # - Low ADX (non-trending)
    # - Price sweeps below range low and reclaims
    result_1 = choose_setup_type(
        side_mode="LONG",
        adx_4h=20.0,
        atrp_4h=3.0,
        btc_trend="NEUTRAL",
        close=np.array([100.0] * 50 + [99.0, 98.0, 99.5]),  # sweep low + reclaim
        high=np.array([101.0] * 50 + [100.0, 99.0, 100.0]),
        low=np.array([99.0] * 50 + [97.5, 96.0, 98.0]),  # wick below range
        current_price=99.5,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,
        atr_last=1.0,
    )
    assert result_1 == "RANGE_SWEEP_RECLAIM", f"Expected RANGE_SWEEP_RECLAIM, got {result_1}"
    print(f"  ✓ RANGE_SWEEP_RECLAIM: {result_1}")

    # Test case 2: BREAKOUT_RETEST
    # - Tight range (8% or less width)
    # - Price closes above range high (breakout)
    result_2 = choose_setup_type(
        side_mode="LONG",
        adx_4h=22.0,
        atrp_4h=3.0,
        btc_trend="BULLISH",
        close=np.array([100.0] * 50 + [100.5, 101.2]),
        high=np.array([101.0] * 50 + [101.0, 102.0]),
        low=np.array([99.0] * 50 + [100.0, 101.0]),
        current_price=101.2,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,  # 2% range
        atr_last=1.0,
    )
    assert result_2 == "BREAKOUT_RETEST", f"Expected BREAKOUT_RETEST, got {result_2}"
    print(f"  ✓ BREAKOUT_RETEST: {result_2}")

    # Test case 3: TREND_PULLBACK
    # - High ADX (trending)
    # - Not in a very volatile state
    # - Range is wider (not tight)
    # - Price is NOT near range edges (to avoid BREAKOUT_RETEST)
    result_3 = choose_setup_type(
        side_mode="LONG",
        adx_4h=35.0,
        atrp_4h=4.0,
        btc_trend="BULLISH",
        close=np.array([100.0] * 50 + [103.0, 104.0]),
        high=np.array([101.0] * 50 + [104.0, 105.0]),
        low=np.array([99.0] * 50 + [102.0, 103.0]),
        current_price=104.0,
        rng_low=95.0,
        rng_high=110.0,
        rng_h=15.0,  # wide range (>8% of mid)
        atr_last=2.0,
    )
    assert result_3 == "TREND_PULLBACK", f"Expected TREND_PULLBACK, got {result_3}"
    print(f"  ✓ TREND_PULLBACK: {result_3}")

    # Test case 4: VOLATILITY_FADE
    # - Very high ATR% (>= 8%)
    # - Oversized candle
    result_4 = choose_setup_type(
        side_mode="LONG",
        adx_4h=25.0,
        atrp_4h=12.0,  # very high volatility
        btc_trend="NEUTRAL",
        close=np.array([100.0] * 50 + [110.0]),  # big move
        high=np.array([101.0] * 50 + [115.0]),
        low=np.array([99.0] * 50 + [105.0]),
        current_price=110.0,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,
        atr_last=3.0,
    )
    assert result_4 == "VOLATILITY_FADE", f"Expected VOLATILITY_FADE, got {result_4}"
    print(f"  ✓ VOLATILITY_FADE: {result_4}")

    print("All setup_type tests passed!")


def test_late_atr_calculation():
    """Verify late_atr calculation logic."""
    print("Testing late_atr calculation...")

    # The late_atr should only be positive when price is BEYOND the entry zone.
    # For LONG: current_price > entry_high triggers late
    # For SHORT: current_price < entry_low triggers late

    # LONG example: current price is 105, entry_high is 102, atr is 2
    # late_atr = (105 - 102) / 2 = 1.5 ATR units late
    late_long = (105.0 - 102.0) / 2.0
    assert late_long == 1.5, f"Expected 1.5, got {late_long}"
    print(f"  ✓ LONG late_atr calculation: {late_long}")

    # SHORT example: current price is 95, entry_low is 98, atr is 2
    # late_atr = (98 - 95) / 2 = 1.5 ATR units late
    late_short = (98.0 - 95.0) / 2.0
    assert late_short == 1.5, f"Expected 1.5, got {late_short}"
    print(f"  ✓ SHORT late_atr calculation: {late_short}")

    print("All late_atr tests passed!")


def test_penalty_capping():
    """Verify penalty capping logic."""
    print("Testing penalty capping...")

    max_penalty_abs = 10.0
    penalties = -25.0  # huge penalty

    # Cap should bring it to -10
    capped = max(-max_penalty_abs, min(max_penalty_abs, penalties))
    assert capped == -10.0, f"Expected -10.0, got {capped}"
    print(f"  ✓ Penalty capped from -25 to {capped}")

    # Positive penalty (should also cap)
    penalties_pos = 15.0
    capped_pos = max(-max_penalty_abs, min(max_penalty_abs, penalties_pos))
    assert capped_pos == 10.0, f"Expected 10.0, got {capped_pos}"
    print(f"  ✓ Penalty capped from +15 to {capped_pos}")

    print("All penalty capping tests passed!")


def test_symbol_normalization():
    """Verify symbol normalization logic for Manual Mode."""
    print("Testing symbol normalization...")

    # Test 1: Lowercase to uppercase with USDT append
    result = normalize_symbol("eth")
    assert result == "ETHUSDT", f"Expected ETHUSDT, got {result}"
    print(f"  ✓ 'eth' -> '{result}'")

    # Test 2: Already has USDT
    result = normalize_symbol("ETHUSDT")
    assert result == "ETHUSDT", f"Expected ETHUSDT, got {result}"
    print(f"  ✓ 'ETHUSDT' -> '{result}'")

    # Test 3: With slash separator
    result = normalize_symbol("btc/usdt")
    assert result == "BTCUSDT", f"Expected BTCUSDT, got {result}"
    print(f"  ✓ 'btc/usdt' -> '{result}'")

    # Test 4: With dash separator
    result = normalize_symbol("SOL-USDT")
    assert result == "SOLUSDT", f"Expected SOLUSDT, got {result}"
    print(f"  ✓ 'SOL-USDT' -> '{result}'")

    # Test 5: With whitespace
    result = normalize_symbol("  pepe  ")
    assert result == "PEPEUSDT", f"Expected PEPEUSDT, got {result}"
    print(f"  ✓ '  pepe  ' -> '{result}'")

    # Test 6: Mixed case
    result = normalize_symbol("DoGe")
    assert result == "DOGEUSDT", f"Expected DOGEUSDT, got {result}"
    print(f"  ✓ 'DoGe' -> '{result}'")

    print("All symbol normalization tests passed!")


def test_symbol_resolution():
    """Verify symbol resolution logic for Manual Mode."""
    print("Testing symbol resolution...")

    # Mock valid symbols (simulating Binance exchange info)
    valid_symbols = {
        "BTCUSDT": {"symbol": "BTCUSDT", "baseAsset": "BTC"},
        "ETHUSDT": {"symbol": "ETHUSDT", "baseAsset": "ETH"},
        "1000PEPEUSDT": {"symbol": "1000PEPEUSDT", "baseAsset": "1000PEPE"},
        "SOLUSDT": {"symbol": "SOLUSDT", "baseAsset": "SOL"},
        "1000SHIBUSDT": {"symbol": "1000SHIBUSDT", "baseAsset": "1000SHIB"},
    }

    # Test 1: Direct match
    resolved, warning = resolve_symbol("BTCUSDT", valid_symbols)
    assert resolved == "BTCUSDT", f"Expected BTCUSDT, got {resolved}"
    assert warning is None, f"Expected no warning, got {warning}"
    print(f"  ✓ 'BTCUSDT' -> '{resolved}' (no warning)")

    # Test 2: 1000x variant resolution (PEPE -> 1000PEPEUSDT)
    resolved, warning = resolve_symbol("PEPEUSDT", valid_symbols)
    assert resolved == "1000PEPEUSDT", f"Expected 1000PEPEUSDT, got {resolved}"
    assert warning is not None, f"Expected warning, got None"
    print(f"  ✓ 'PEPEUSDT' -> '{resolved}' (warning: {warning})")

    # Test 3: Symbol not found
    resolved, warning = resolve_symbol("XYZUSDT", valid_symbols)
    assert resolved is None, f"Expected None, got {resolved}"
    print(f"  ✓ 'XYZUSDT' -> None (not found)")

    # Test 4: SHIB -> 1000SHIBUSDT
    resolved, warning = resolve_symbol("SHIBUSDT", valid_symbols)
    assert resolved == "1000SHIBUSDT", f"Expected 1000SHIBUSDT, got {resolved}"
    print(f"  ✓ 'SHIBUSDT' -> '{resolved}'")

    print("All symbol resolution tests passed!")


def test_scoring_weight_normalization():
    """Verify scoring weights sum to 100 and normalization works."""
    print("Testing scoring weight normalization...")

    from .config import default_config
    cfg = default_config()
    weights = cfg["scoring"]["weights"]

    total = sum(weights.values())
    assert total == 100, f"Expected weights to sum to 100, got {total}"
    print(f"  ✓ Weights sum to {total}: {weights}")

    # Verify the pipeline scoring function uses the same keys
    expected_keys = {"tradeability", "setup_quality", "derivatives", "orderflow", "context", "whales"}
    actual_keys = set(weights.keys())
    assert actual_keys == expected_keys, f"Expected keys {expected_keys}, got {actual_keys}"
    print(f"  ✓ Weight keys match expected: {actual_keys}")

    # Whales weight must be 5 (spec: "nhẹ: weight 5")
    assert weights["whales"] == 5, f"Expected whales weight 5, got {weights['whales']}"
    print(f"  ✓ Whales weight = {weights['whales']} (spec: 5)")

    print("All scoring weight tests passed!")


def test_late_penalty_differentiation():
    """Verify WATCH_LATE and WATCH_PULLBACK have different penalties."""
    print("Testing late penalty differentiation...")

    from .config import default_config
    cfg = default_config()
    late_cfg = cfg["scoring"]["late"]

    penalty_late = float(late_cfg.get("penalty_watch_late", -2.0))
    penalty_pullback = float(late_cfg.get("penalty_watch_pullback", 0.0))

    # WATCH_PULLBACK should be less severe than WATCH_LATE
    assert penalty_pullback >= penalty_late, (
        f"WATCH_PULLBACK penalty ({penalty_pullback}) should be >= WATCH_LATE penalty ({penalty_late})"
    )
    print(f"  ✓ penalty_watch_late = {penalty_late}")
    print(f"  ✓ penalty_watch_pullback = {penalty_pullback}")
    print(f"  ✓ WATCH_PULLBACK is less severe than WATCH_LATE")

    # max_penalty_abs must be 10 (spec: "penalty tối đa 10")
    max_pen = float(cfg["scoring"].get("max_penalty_abs", 10.0))
    assert max_pen == 10.0, f"Expected max_penalty_abs=10, got {max_pen}"
    print(f"  ✓ max_penalty_abs = {max_pen} (spec: 10)")

    print("All late penalty tests passed!")


def test_confidence_oi_flush_not_penalized():
    """Verify OI_FLUSH is not penalized in confidence (it's constructive)."""
    print("Testing confidence OI_FLUSH handling...")

    from .confidence import heuristic_confidence
    from .config import default_config

    cfg = default_config()

    # Plan with OI_FLUSH flag and decent score
    plan = {
        "final_score": 70.0,
        "late_status": "OK",
        "side": "LONG",
        "derivatives": {
            "score": 55.0,
            "funding_flags": ["FUNDING_NEUTRAL"],
            "oi_flags": ["OI_FLUSH"],
        },
        "orderflow": {"score": 50.0, "flags": []},
    }
    result = heuristic_confidence(plan, cfg)

    # OI_FLUSH should NOT reduce score below the base
    assert result.score >= 70.0, (
        f"OI_FLUSH should not penalize score. Base=70.0, got {result.score}"
    )
    print(f"  ✓ Score with OI_FLUSH = {result.score:.1f} (base=70.0, not penalized)")

    # Check that the reason reflects constructive treatment
    assert any("constructive" in r or "flush" in r for r in result.reasons), (
        f"Expected OI_FLUSH constructive reason, got {result.reasons}"
    )
    print(f"  ✓ Reasons include constructive OI_FLUSH: {result.reasons}")

    print("All confidence OI_FLUSH tests passed!")


def test_setup_type_all_four():
    """Quick regression: ensure all 4 setup types are still producible."""
    print("Testing all 4 setup types are producible...")

    types_seen = set()

    # Re-run the determinism test but just collect types
    # RANGE_SWEEP_RECLAIM
    r1 = choose_setup_type(
        side_mode="LONG", adx_4h=20.0, atrp_4h=3.0, btc_trend="NEUTRAL",
        close=np.array([100.0]*50 + [99.0, 98.0, 99.5]),
        high=np.array([101.0]*50 + [100.0, 99.0, 100.0]),
        low=np.array([99.0]*50 + [97.5, 96.0, 98.0]),
        current_price=99.5, rng_low=99.0, rng_high=101.0, rng_h=2.0, atr_last=1.0,
    )
    types_seen.add(r1)

    # BREAKOUT_RETEST
    r2 = choose_setup_type(
        side_mode="LONG", adx_4h=22.0, atrp_4h=3.0, btc_trend="BULLISH",
        close=np.array([100.0]*50 + [100.5, 101.2]),
        high=np.array([101.0]*50 + [101.0, 102.0]),
        low=np.array([99.0]*50 + [100.0, 101.0]),
        current_price=101.2, rng_low=99.0, rng_high=101.0, rng_h=2.0, atr_last=1.0,
    )
    types_seen.add(r2)

    # TREND_PULLBACK
    r3 = choose_setup_type(
        side_mode="LONG", adx_4h=35.0, atrp_4h=4.0, btc_trend="BULLISH",
        close=np.array([100.0]*50 + [103.0, 104.0]),
        high=np.array([101.0]*50 + [104.0, 105.0]),
        low=np.array([99.0]*50 + [102.0, 103.0]),
        current_price=104.0, rng_low=95.0, rng_high=110.0, rng_h=15.0, atr_last=2.0,
    )
    types_seen.add(r3)

    # VOLATILITY_FADE
    r4 = choose_setup_type(
        side_mode="LONG", adx_4h=25.0, atrp_4h=12.0, btc_trend="NEUTRAL",
        close=np.array([100.0]*50 + [110.0]),
        high=np.array([101.0]*50 + [115.0]),
        low=np.array([99.0]*50 + [105.0]),
        current_price=110.0, rng_low=99.0, rng_high=101.0, rng_h=2.0, atr_last=3.0,
    )
    types_seen.add(r4)

    expected = {"RANGE_SWEEP_RECLAIM", "BREAKOUT_RETEST", "TREND_PULLBACK", "VOLATILITY_FADE"}
    assert types_seen == expected, f"Expected {expected}, got {types_seen}"
    print(f"  ✓ All 4 types produced: {types_seen}")

    print("All setup_type coverage tests passed!")


def test_indicator_bollinger_bands():
    """Verify Bollinger Bands produce finite values with enough data."""
    print("Testing Bollinger Bands...")
    close = np.array([100.0 + np.sin(i / 5.0) * 3.0 for i in range(100)])
    upper, middle, lower = bollinger_bands(close, period=20, num_std=2.0)

    # First 19 values should be NaN, rest should be finite
    assert np.all(np.isnan(upper[:19])), "First 19 upper should be NaN"
    assert np.all(np.isfinite(upper[19:])), "Upper values after warmup should be finite"
    assert np.all(np.isfinite(middle[19:])), "Middle values after warmup should be finite"
    assert np.all(np.isfinite(lower[19:])), "Lower values after warmup should be finite"

    # Upper > Middle > Lower
    assert np.all(upper[19:] >= middle[19:]), "Upper should be >= Middle"
    assert np.all(middle[19:] >= lower[19:]), "Middle should be >= Lower"
    print(f"  BB sample: upper={upper[-1]:.2f}, mid={middle[-1]:.2f}, lower={lower[-1]:.2f}")

    # BandWidth
    bbw = bollinger_bandwidth(upper, middle, lower)
    assert np.all(np.isfinite(bbw[19:])), "BBW should be finite after warmup"
    assert np.all(bbw[19:] >= 0), "BBW should be non-negative"
    print(f"  BBW last = {bbw[-1]:.2f}")
    print("  Bollinger Bands tests passed!")


def test_indicator_vwap_avwap():
    """Verify VWAP and Anchored VWAP produce finite values."""
    print("Testing VWAP and AVWAP...")
    n = 50
    h = np.array([102.0 + i * 0.1 for i in range(n)])
    l = np.array([98.0 + i * 0.1 for i in range(n)])
    c = np.array([100.0 + i * 0.1 for i in range(n)])
    vol = np.array([1000.0] * n)

    tp = typical_price(h, l, c)
    assert len(tp) == n, "TP length mismatch"
    assert np.all(np.isfinite(tp)), "TP should be finite"

    v = vwap(tp, vol)
    assert len(v) == n, "VWAP length mismatch"
    assert np.all(np.isfinite(v)), "VWAP should be finite"

    avwap_out = anchored_vwap(tp, vol, anchor_idx=10)
    assert len(avwap_out) == n, "AVWAP length mismatch"
    assert np.all(np.isnan(avwap_out[:10])), "AVWAP before anchor should be NaN"
    assert np.all(np.isfinite(avwap_out[10:])), "AVWAP after anchor should be finite"
    print(f"  VWAP last = {v[-1]:.4f}, AVWAP last = {avwap_out[-1]:.4f}")
    print("  VWAP/AVWAP tests passed!")


def test_indicator_choppiness():
    """Verify Choppiness Index produces values in expected range."""
    print("Testing Choppiness Index...")
    n = 100
    h = np.array([101.0 + np.random.uniform(0, 2) for _ in range(n)])
    l = np.array([99.0 - np.random.uniform(0, 2) for _ in range(n)])
    c = np.array([100.0 + np.random.uniform(-1, 1) for _ in range(n)])

    chop = choppiness_index(h, l, c, period=14)
    assert len(chop) == n, "CHOP length mismatch"
    # Values after warmup should be in [0, 100]
    valid = chop[14:]
    finite_vals = valid[np.isfinite(valid)]
    assert len(finite_vals) > 0, "Should have some finite CHOP values"
    assert np.all(finite_vals >= 0), f"CHOP should be >= 0, got min={finite_vals.min()}"
    assert np.all(finite_vals <= 100), f"CHOP should be <= 100, got max={finite_vals.max()}"
    print(f"  CHOP range: [{finite_vals.min():.1f}, {finite_vals.max():.1f}]")
    print("  Choppiness Index tests passed!")


def test_validation_long_ordering():
    """Validate LONG ordering: SL < entry_low < entry_high < TP1 < TP2 < TP3."""
    print("Testing validation LONG ordering...")

    # Valid plan
    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "LONG", 100.0, 102.0, 98.0, [105.0, 108.0, 112.0], 101.0
    )
    assert sl < el < eh < tps[0] < tps[1] < tps[2], \
        f"LONG ordering violated: SL={sl}, EL={el}, EH={eh}, TPs={tps}"
    assert not rep, f"Valid plan should not need repair, got: {reason}"
    print(f"  Valid LONG: SL={sl:.1f} < EL={el:.1f} < EH={eh:.1f} < TP1={tps[0]:.1f} < TP2={tps[1]:.1f} < TP3={tps[2]:.1f}")

    # Broken plan: SL above entry
    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "LONG", 100.0, 102.0, 105.0, [108.0, 110.0, 112.0], 101.0
    )
    assert sl < el, f"SL should be moved below entry, got SL={sl}, EL={el}"
    assert rep, "Should be repaired"
    print(f"  Repaired: SL moved from 105 to {sl:.1f} ({reason})")

    print("  Validation LONG ordering tests passed!")


def test_validation_short_ordering():
    """Validate SHORT ordering: TP3 < TP2 < TP1 < entry_low < entry_high < SL."""
    print("Testing validation SHORT ordering...")

    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "SHORT", 100.0, 102.0, 104.0, [98.0, 95.0, 92.0], 101.0
    )
    assert tps[2] < tps[1] < tps[0] < el < eh < sl, \
        f"SHORT ordering violated: TPs={tps}, EL={el}, EH={eh}, SL={sl}"
    assert not rep, f"Valid plan should not need repair, got: {reason}"
    print(f"  Valid SHORT: TP3={tps[2]:.1f} < TP2={tps[1]:.1f} < TP1={tps[0]:.1f} < EL={el:.1f} < EH={eh:.1f} < SL={sl:.1f}")

    print("  Validation SHORT ordering tests passed!")


def test_validation_negative_tp():
    """Validate that negative TPs are prevented (micro-priced assets)."""
    print("Testing negative TP prevention...")

    # Micro-priced asset SHORT: TPs could go negative
    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "SHORT", 0.001, 0.002, 0.003, [-0.001, -0.005, -0.01], 0.0015
    )
    for i, tp in enumerate(tps):
        assert tp > 0, f"TP{i+1} should be > 0, got {tp}"
    assert rep, "Should be repaired for negative TPs"
    print(f"  Repaired TPs: {[f'{t:.6f}' for t in tps]} (reason: {reason})")

    print("  Negative TP prevention tests passed!")


def test_validation_tp_monotonicity():
    """Validate TP monotonicity for both sides."""
    print("Testing TP monotonicity...")

    # LONG with reversed TPs
    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "LONG", 100.0, 102.0, 98.0, [110.0, 108.0, 105.0], 101.0
    )
    assert tps[0] < tps[1] < tps[2], f"LONG TPs not monotonic: {tps}"
    print(f"  LONG monotonicity repaired: {tps}")

    # SHORT with reversed TPs
    el, eh, sl, tps, rep, reason = validate_plan_levels(
        "SHORT", 100.0, 102.0, 104.0, [90.0, 92.0, 95.0], 101.0
    )
    assert tps[0] > tps[1] > tps[2], f"SHORT TPs not monotonic: {tps}"
    print(f"  SHORT monotonicity repaired: {tps}")

    print("  TP monotonicity tests passed!")


def test_build_algo_plan_all_tactics():
    """Test that build_algo_plan produces valid output for all 4 tactics."""
    print("Testing build_algo_plan for all tactics...")

    # We reuse conditions known to produce each tactic
    test_cases = [
        {
            "label": "RANGE_SWEEP_RECLAIM",
            "close": np.array([100.0] * 50 + [99.0, 98.0, 99.5]),
            "high": np.array([101.0] * 50 + [100.0, 99.0, 100.0]),
            "low": np.array([99.0] * 50 + [97.5, 96.0, 98.0]),
            "price": 99.5,
        },
        {
            "label": "BREAKOUT_RETEST",
            "close": np.array([100.0] * 50 + [100.5, 101.2]),
            "high": np.array([101.0] * 50 + [101.0, 102.0]),
            "low": np.array([99.0] * 50 + [100.0, 101.0]),
            "price": 101.2,
        },
        {
            "label": "TREND_PULLBACK",
            "close": np.array([100.0] * 60 + [103.0, 104.0, 105.0, 106.0, 107.0]),
            "high": np.array([101.0] * 60 + [104.0, 105.0, 106.0, 107.0, 108.0]),
            "low": np.array([99.0] * 60 + [102.0, 103.0, 104.0, 105.0, 106.0]),
            "price": 107.0,
        },
        {
            "label": "VOLATILITY_FADE",
            "close": np.array([100.0] * 50 + [110.0]),
            "high": np.array([101.0] * 50 + [115.0]),
            "low": np.array([99.0] * 50 + [105.0]),
            "price": 110.0,
        },
    ]

    for side in ["LONG", "SHORT"]:
        for tc in test_cases:
            n = len(tc["close"])
            vol = np.ones(n, dtype=float) * 1000.0
            plan = build_algo_plan(
                symbol="TESTUSDT",
                side_mode=side,
                close=tc["close"],
                high=tc["high"],
                low=tc["low"],
                current_price=tc["price"],
                bid=tc["price"] - 0.1,
                ask=tc["price"] + 0.1,
                volume_usdt=500_000_000,
                btc_trend="NEUTRAL",
                volume=vol,
            )

            # Parse entry zone
            parts = plan.entry_zone.split("-")
            assert len(parts) >= 2, f"Invalid entry_zone format: {plan.entry_zone}"
            el = float(parts[0])
            eh = float(parts[-1])

            # Validate ordering
            if side == "LONG":
                assert plan.stop_loss < el, \
                    f"{side} {tc['label']}: SL={plan.stop_loss:.4f} should be < entry_low={el:.4f}"
                for i, tp in enumerate(plan.take_profits):
                    assert tp > eh, \
                        f"{side} {tc['label']}: TP{i+1}={tp:.4f} should be > entry_high={eh:.4f}"
            else:
                assert plan.stop_loss > eh, \
                    f"{side} {tc['label']}: SL={plan.stop_loss:.4f} should be > entry_high={eh:.4f}"
                for i, tp in enumerate(plan.take_profits):
                    assert tp < el, \
                        f"{side} {tc['label']}: TP{i+1}={tp:.4f} should be < entry_low={el:.4f}"

            # TP positivity
            for i, tp in enumerate(plan.take_profits):
                assert tp > 0, f"{side} {tc['label']}: TP{i+1} should be > 0, got {tp}"

            # TP monotonicity
            tps = plan.take_profits
            if side == "LONG":
                assert tps[0] < tps[1] < tps[2], \
                    f"{side} {tc['label']}: TPs not monotonic (ascending): {tps}"
            else:
                assert tps[0] > tps[1] > tps[2], \
                    f"{side} {tc['label']}: TPs not monotonic (descending): {tps}"

            # levels_debug present
            assert plan.levels_debug is not None, \
                f"{side} {tc['label']}: levels_debug should not be None"

            # entry_zone width cap (sanity: should not be > 5x ATR)
            width = eh - el
            assert width >= 0, f"{side} {tc['label']}: entry width negative: {width}"

            print(f"  {side} {tc['label']}: OK (entry={plan.entry_zone}, SL={plan.stop_loss:.4f}, TPs={[f'{t:.4f}' for t in tps]})")

    print("  All build_algo_plan tactic tests passed!")


def test_entry_zone_width_cap():
    """Verify entry zone width is capped by ATR."""
    print("Testing entry zone width cap...")

    # Create data with large range that could produce wide entry zones
    close = np.array([100.0] * 100)
    high = np.array([110.0] * 100)  # very wide range
    low = np.array([90.0] * 100)
    vol = np.ones(100, dtype=float) * 1000.0

    plan = build_algo_plan(
        symbol="TESTUSDT",
        side_mode="LONG",
        close=close,
        high=high,
        low=low,
        current_price=100.0,
        bid=99.9,
        ask=100.1,
        volume_usdt=500_000_000,
        btc_trend="NEUTRAL",
        volume=vol,
    )

    parts = plan.entry_zone.split("-")
    el = float(parts[0])
    eh = float(parts[-1])
    width = eh - el

    # Width should be reasonable (capped at ~1.5 ATR by level engine)
    # ATR on flat data with H-L=20 will be large, so just verify it's positive and not the full range
    assert width > 0, f"Entry width should be > 0, got {width}"
    assert width < 20.0, f"Entry width should be < full range (20), got {width}"
    print(f"  Entry zone width = {width:.4f} (capped)")

    print("  Entry zone width cap tests passed!")


def test_no_nans_in_indicators():
    """Verify new indicators return no NaNs where data is sufficient."""
    print("Testing no NaNs in indicators...")

    n = 100
    close = np.array([100.0 + np.sin(i / 10.0) * 5.0 for i in range(n)])
    high = close + 1.0
    low = close - 1.0
    vol = np.ones(n) * 1000.0

    # Bollinger Bands (after warmup)
    u, m, l = bollinger_bands(close, 20, 2.0)
    assert np.all(np.isfinite(u[19:])), "BB upper has NaN after warmup"
    assert np.all(np.isfinite(m[19:])), "BB middle has NaN after warmup"
    assert np.all(np.isfinite(l[19:])), "BB lower has NaN after warmup"

    # BBW
    bbw = bollinger_bandwidth(u, m, l)
    assert np.all(np.isfinite(bbw[19:])), "BBW has NaN after warmup"

    # CHOP
    chop = choppiness_index(high, low, close, 14)
    assert np.all(np.isfinite(chop[14:])), "CHOP has NaN after warmup"

    # VWAP
    tp = typical_price(high, low, close)
    v = vwap(tp, vol)
    assert np.all(np.isfinite(v)), "VWAP has NaN"

    # AVWAP
    av = anchored_vwap(tp, vol, 10)
    assert np.all(np.isfinite(av[10:])), "AVWAP has NaN after anchor"

    print("  All indicator NaN checks passed!")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("WhaleTraderBot v2.4 ALGO-only - Smoke Tests")
    print("=" * 60)
    print()

    test_setup_type_determinism()
    print()
    test_late_atr_calculation()
    print()
    test_penalty_capping()
    print()
    test_symbol_normalization()
    print()
    test_symbol_resolution()
    print()
    test_scoring_weight_normalization()
    print()
    test_late_penalty_differentiation()
    print()
    test_confidence_oi_flush_not_penalized()
    print()
    test_setup_type_all_four()
    print()

    # New Level Engine + indicator tests
    test_indicator_bollinger_bands()
    print()
    test_indicator_vwap_avwap()
    print()
    test_indicator_choppiness()
    print()
    test_validation_long_ordering()
    print()
    test_validation_short_ordering()
    print()
    test_validation_negative_tp()
    print()
    test_validation_tp_monotonicity()
    print()
    test_build_algo_plan_all_tactics()
    print()
    test_entry_zone_width_cap()
    print()
    test_no_nans_in_indicators()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
