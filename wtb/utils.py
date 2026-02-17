from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover
    repair_json = None  # type: ignore


def utc_now_iso() -> str:
    """UTC timestamp (timezone-aware)."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_default(obj: Any) -> Any:
    """Make common scientific/python objects JSON serializable."""
    # numpy
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # pandas
    if pd is not None:
        if isinstance(obj, getattr(pd, "Timestamp")):
            return obj.to_pydatetime().replace(tzinfo=dt.timezone.utc).isoformat()
        if isinstance(obj, getattr(pd, "Timedelta")):
            return str(obj)
    # datetime
    if isinstance(obj, (dt.datetime, dt.date)):
        if isinstance(obj, dt.datetime) and obj.tzinfo is None:
            obj = obj.replace(tzinfo=dt.timezone.utc)
        return obj.isoformat()
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    # fallback
    return str(obj)


def json_dumps(data: Any, pretty: bool = False) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        default=json_default,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


JSONSpan = Tuple[str, int, int]


def _find_json_spans(text: str) -> list[JSONSpan]:
    """Return candidate JSON object/array spans in the text."""
    spans: list[JSONSpan] = []
    for m in re.finditer(r"\[", text):
        start = m.start()
        sub = text[start:]
        end = _match_brackets(sub, open_ch="[", close_ch="]")
        if end is not None:
            spans.append((text[start:start + end], start, start + end))
    for m in re.finditer(r"\{", text):
        start = m.start()
        sub = text[start:]
        end = _match_brackets(sub, open_ch="{", close_ch="}")
        if end is not None:
            spans.append((text[start:start + end], start, start + end))
    # sort by length descending (prefer largest)
    spans.sort(key=lambda x: len(x[0]), reverse=True)
    return spans


def _match_brackets(s: str, open_ch: str, close_ch: str) -> Optional[int]:
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json_from_text(text: str) -> Any:
    """Extract and parse JSON from a model response.

    Strategy:
    1) Try direct json.loads.
    2) Try extracting the largest JSON array/object substring.
    3) If still fails and json-repair is installed, repair then parse.
    """
    text = text.strip()
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) spans
    spans = _find_json_spans(text)
    for blob, _, _ in spans:
        try:
            return json.loads(blob)
        except Exception:
            continue

    # 3) repair
    if repair_json is not None:
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            pass

    raise ValueError("No JSON found in model output.")


@dataclass
class Timer:
    start: float

    @classmethod
    def begin(cls) -> "Timer":
        import time
        return cls(start=time.time())

    def ms(self) -> float:
        import time
        return (time.time() - self.start) * 1000.0


@dataclass
class RateLimiter:
    """Simple token-bucket style limiter for HTTP endpoints."""

    min_interval_sec: float
    _last_ts: float = 0.0

    def wait(self) -> None:
        import time as _time
        now = _time.time()
        elapsed = now - self._last_ts
        if elapsed < self.min_interval_sec:
            _time.sleep(self.min_interval_sec - elapsed)
        self._last_ts = _time.time()
