from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ConfidenceResult:
    score: float
    label: str  # EXECUTE | WATCH | SKIP
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "label": self.label, "reasons": self.reasons}


def heuristic_confidence(plan: Dict[str, Any], cfg: Dict[str, Any]) -> ConfidenceResult:
    """Heuristic confidence gate.

    This is intentionally *simple* but robust:
    - Base = final_score
    - Penalize late / extreme derivatives / obvious orderflow conflict
    - Apply external_intel multiplier (if enabled)
    """
    conf_cfg = cfg.get("confidence", {})
    enabled = bool(conf_cfg.get("enabled", True))
    if not enabled:
        return ConfidenceResult(score=float(plan.get("final_score", 0.0)), label="WATCH", reasons=["confidence_disabled"])

    base = float(plan.get("final_score", 0.0) or 0.0)
    score = base
    reasons: List[str] = []

    # Late penalty
    late_status = str(plan.get("late_status", ""))
    if late_status in ("WATCH_LATE", "WATCH_PULLBACK"):
        score -= float(conf_cfg.get("late_penalty", 5.0))
        reasons.append(f"late:{late_status}")

    # Derivatives penalties
    deriv = plan.get("derivatives") or {}
    if isinstance(deriv, dict):
        flags = set((deriv.get("funding_flags") or []) + (deriv.get("oi_flags") or []))
        if "FUNDING_EXTREME" in flags:
            score -= 7.0
            reasons.append("funding_extreme")
        if "OI_SPIKE" in flags:
            score -= 4.0
            reasons.append("oi_spike")
        # OI_FLUSH is a constructive signal (liquidation clears crowded positions,
        # aligns with sweep/reclaim philosophy). No penalty; small boost instead.
        if "OI_FLUSH" in flags:
            score += 1.0
            reasons.append("oi_flush_constructive")

    # Orderflow penalties
    of = plan.get("orderflow") or {}
    if isinstance(of, dict):
        of_flags = set(of.get("flags") or [])
        # Extremely naive conflict check, but helps avoid trap entries
        if plan.get("side") == "LONG" and "CVD_DOWN" in of_flags:
            score -= 4.0
            reasons.append("orderflow_cvd_down")
        if plan.get("side") == "SHORT" and "CVD_UP" in of_flags:
            score -= 4.0
            reasons.append("orderflow_cvd_up")

    # External intel multiplier
    ext = plan.get("external_intel") or {}
    if isinstance(ext, dict):
        action = str(ext.get("action", "proceed")).lower()
        mult = float(ext.get("multiplier", 1.0) or 1.0)
        if action in ("skip", "pause_trading_24h"):
            score = 0.0
            reasons.append(f"external_veto:{action}")
        else:
            if bool(conf_cfg.get("apply_external_multiplier", True)):
                score *= mult
                if mult < 1.0:
                    reasons.append(f"external_mult:{mult:.2f}")

    # Clamp
    score = float(max(0.0, min(100.0, score)))

    # Thresholds (mode can override these upstream)
    min_execute = float(conf_cfg.get("min_execute", 75))
    min_watch = float(conf_cfg.get("min_watch", 65))

    if score >= min_execute:
        label = "EXECUTE"
    elif score >= min_watch:
        label = "WATCH"
    else:
        label = "SKIP"

    if not reasons:
        reasons = ["clean"]
    return ConfidenceResult(score=score, label=label, reasons=reasons)
