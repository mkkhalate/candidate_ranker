"""
utils.py — Logging, timing, and memory monitoring utilities.
"""

import time
import json
import psutil
import os
from datetime import datetime
from typing import Any, Dict, Optional


_timings: Dict[str, float] = {}
_step_starts: Dict[str, float] = {}


def ts() -> str:
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log(step: str, message: str, level: str = "INFO") -> None:
    """Print a detailed timestamped log line."""
    print(f"[{ts()}] [{level}] [{step}] {message}", flush=True)


def log_memory() -> None:
    """Print current RAM usage."""
    proc = psutil.Process(os.getpid())
    used_gb = proc.memory_info().rss / (1024 ** 3)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    pct = (used_gb / total_gb) * 100
    print(
        f"[{ts()}] [MEMORY] Current RAM usage: {used_gb:.2f} GB / {total_gb:.1f} GB ({pct:.1f}%)",
        flush=True,
    )


def start_timer(key: str) -> None:
    """Start timing a named step."""
    _step_starts[key] = time.perf_counter()
    log(key, f"Starting...")


def stop_timer(key: str, extra: str = "") -> float:
    """Stop timing a named step and record it."""
    elapsed = time.perf_counter() - _step_starts.get(key, time.perf_counter())
    _timings[key] = elapsed
    msg = f"Completed in {elapsed:.4f}s"
    if extra:
        msg += f" — {extra}"
    log(key, msg)
    return elapsed


def record_timing(key: str, elapsed: float) -> None:
    """Record a timing directly."""
    _timings[key] = elapsed


def save_timings(output_dir: str) -> None:
    """Write execution_timings.json to the output directory."""
    path = os.path.join(output_dir, "execution_timings.json")
    with open(path, "w") as f:
        json.dump(_timings, f, indent=2)
    log("TIMINGS", f"Saved execution timings to {path}")


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    try:
        return os.path.getsize(path) / (1024 ** 2)
    except OSError:
        return 0.0


def progress_bar(current: int, total: int, bar_width: int = 20) -> str:
    """Return a simple ASCII progress bar string."""
    frac = current / total if total > 0 else 0
    filled = int(bar_width * frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = frac * 100
    return f"{pct:5.1f}%|{bar}| {current}/{total}"
