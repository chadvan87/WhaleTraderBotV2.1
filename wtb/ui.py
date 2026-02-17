from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


def build_progress(console: Console, refresh_per_second: int = 20) -> Progress:
    """Standardized Rich progress layout used across pipeline stages."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=refresh_per_second,
    )


@contextmanager
def progress_context(console: Console, enabled: bool = True, refresh_per_second: int = 20) -> Iterator[Progress | None]:
    """Context manager that yields a Progress if enabled else None."""
    if not enabled:
        yield None
        return
    prog = build_progress(console, refresh_per_second=refresh_per_second)
    with prog:
        yield prog
