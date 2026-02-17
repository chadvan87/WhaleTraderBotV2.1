from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests


class ExternalIntelProvider(ABC):
    """Provider interface for external intelligence.

    IMPORTANT: providers must return JSON-serializable dicts.
    Providers are allowed to "veto" (skip/reduce/pause) but should NEVER
    invent entry/SL/TP.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fetch_sentiment(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def fetch_onchain(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def fetch_events(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


class MockIntelProvider(ExternalIntelProvider):
    """Safe default provider (no external calls). Returns neutral signals."""

    def name(self) -> str:
        return "mock"

    def fetch_sentiment(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "asset": symbol,
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "key_narratives": [],
            "euphoria_risk": "low",
            "action_for_rrt_pb": "proceed",
            "confidence": 50,
            "reasoning": "mock provider (neutral)",
            "ts_utc": int(time.time()),
        }

    def fetch_onchain(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "asset": symbol,
            "net_exchange_flow": "neutral",
            "whale_activity": "neutral",
            "key_metrics_summary": [],
            "distribution_risk": "low",
            "action_for_rrt_pb": "proceed",
            "confidence": 50,
            "reasoning": "mock provider (neutral)",
            "ts_utc": int(time.time()),
        }

    def fetch_events(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "asset": symbol,
            "upcoming_events": [],
            "overall_event_risk": "low",
            "action_for_rrt_pb": "proceed",
            "confidence": 50,
            "reasoning": "mock provider (neutral)",
            "ts_utc": int(time.time()),
        }


class HttpJSONIntelProvider(ExternalIntelProvider):
    """Calls your own HTTP endpoints that wrap Grok / news / onchain / calendar.

    Expected endpoints:
      - sentiment_url
      - onchain_url
      - events_url

    Each endpoint should accept POST JSON and return JSON.

    This provider is intentionally generic so you can swap providers without
    touching the trading logic.
    """

    def __init__(self, sentiment_url: str, onchain_url: str, events_url: str, timeout_sec: int = 30):
        self.sentiment_url = sentiment_url
        self.onchain_url = onchain_url
        self.events_url = events_url
        self.timeout_sec = timeout_sec

    def name(self) -> str:
        return "http"

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(url, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, dict):
            raise ValueError(f"Provider response must be JSON object, got {type(out)}")
        return out

    def fetch_sentiment(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(self.sentiment_url, {"symbol": symbol, "context": context})

    def fetch_onchain(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(self.onchain_url, {"symbol": symbol, "context": context})

    def fetch_events(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(self.events_url, {"symbol": symbol, "context": context})


def build_provider(cfg: Dict[str, Any]) -> ExternalIntelProvider:
    ext = cfg.get("external_intel", {})
    provider = str(ext.get("provider", "mock")).lower()
    if provider == "mock":
        return MockIntelProvider()

    if provider in ("http", "grok_http"):
        http_cfg = ext.get("http", {})
        sentiment_url = http_cfg.get("sentiment_url") or os.getenv("WTB_SENTIMENT_URL")
        onchain_url = http_cfg.get("onchain_url") or os.getenv("WTB_ONCHAIN_URL")
        events_url = http_cfg.get("events_url") or os.getenv("WTB_EVENTS_URL")
        if not (sentiment_url and onchain_url and events_url):
            raise ValueError(
                "external_intel.provider=http requires http.sentiment_url/http.onchain_url/http.events_url "
                "(or env vars WTB_SENTIMENT_URL/WTB_ONCHAIN_URL/WTB_EVENTS_URL)"
            )
        timeout_sec = int(http_cfg.get("timeout_sec", 30))
        return HttpJSONIntelProvider(str(sentiment_url), str(onchain_url), str(events_url), timeout_sec=timeout_sec)

    raise ValueError(f"Unknown external_intel.provider: {provider}")
