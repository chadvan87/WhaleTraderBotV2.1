from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


ACTION_TO_MULT = {
    "proceed": 1.0,
    "reduce_size_50": 0.5,
    "reduce_size_75": 0.25,
    "skip": 0.0,
    "pause_trading_24h": 0.0,
}


def _safe_action(x: Any) -> str:
    a = str(x or "proceed")
    a = a.strip().lower()
    return a if a in ACTION_TO_MULT else "proceed"


@dataclass
class ExternalIntelBundle:
    provider: str
    sentiment: Dict[str, Any]
    onchain: Dict[str, Any]
    events: Dict[str, Any]
    score: float
    action: str
    multiplier: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "sentiment": self.sentiment,
            "onchain": self.onchain,
            "events": self.events,
            "score": self.score,
            "action": self.action,
            "multiplier": self.multiplier,
        }


class ExternalIntelCache:
    """Tiny JSON cache persisted on disk (optional).

    Key: f"{provider}:{symbol}" -> {"ts": int, "value": dict}
    """

    def __init__(self, path: str, ttl_sec: int = 1800):
        self.path = pathlib.Path(path)
        self.ttl_sec = int(ttl_sec)
        self._data: Dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self._data, dict):
                    self._data = {}
        except Exception:
            self._data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        self._load()
        rec = self._data.get(key)
        if not isinstance(rec, dict):
            return None
        ts = int(rec.get("ts", 0) or 0)
        if (int(time.time()) - ts) > self.ttl_sec:
            return None
        val = rec.get("value")
        return val if isinstance(val, dict) else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._load()
        self._data[key] = {"ts": int(time.time()), "value": value}

    def save(self) -> None:
        self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_external_intel(
    sentiment: Dict[str, Any],
    onchain: Dict[str, Any],
    events: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, str, float]:
    """Aggregate the three JSON responses into (score, action, multiplier).

    - score: 0..100 (higher == safer / more aligned)
    - action: proceed / reduce / skip / pause
    - multiplier: 0..1 (position size multiplier)
    """
    w = weights or {"sentiment": 0.40, "onchain": 0.35, "events": 0.25}

    # Convert risk signals into numeric alignment scores.
    # We keep this simple + robust, because provider outputs can vary.
    s_score = float(sentiment.get("confidence", 50) or 50)
    o_score = float(onchain.get("confidence", 50) or 50)
    e_score = float(events.get("confidence", 50) or 50)

    # Allow provider to specify action hints
    s_act = _safe_action(sentiment.get("action_for_rrt_pb"))
    o_act = _safe_action(onchain.get("action_for_rrt_pb"))
    e_act = _safe_action(events.get("action_for_rrt_pb"))

    # If any layer says pause/skip -> veto.
    action = "proceed"
    for a in (s_act, o_act, e_act):
        if a in ("pause_trading_24h",):
            action = "pause_trading_24h"
            break
        if a == "skip":
            action = "skip"

    # If not vetoed, apply the strongest reduction if any
    if action == "proceed":
        if "reduce_size_75" in (s_act, o_act, e_act):
            action = "reduce_size_75"
        elif "reduce_size_50" in (s_act, o_act, e_act):
            action = "reduce_size_50"

    multiplier = float(ACTION_TO_MULT.get(action, 1.0))

    # Score: Weighted confidence, penalized by reduction actions
    raw = (s_score * float(w.get("sentiment", 0.40))) + (o_score * float(w.get("onchain", 0.35))) + (e_score * float(w.get("events", 0.25)))
    # reduction penalty
    if action == "reduce_size_50":
        raw *= 0.85
    elif action == "reduce_size_75":
        raw *= 0.70
    elif action in ("skip", "pause_trading_24h"):
        raw = 0.0

    score = float(max(0.0, min(100.0, raw)))
    return score, action, multiplier
