from __future__ import annotations

import time
from typing import Protocol

import jax

from scm.config import LogLevel


def _format_long_duration(d: float) -> str:
    unit = "s"
    if d > 120:
        d /= 60
        unit = "min"
    if d > 120:
        d /= 60
        unit = "h"
    return f"{d:.1f}{unit}"


class SimLogger(Protocol):
    """Logger interface for `simulate`.

    All methods that receive JAX tracers (`on_outer_step`) must dispatch through
    `jax.debug.callback` so that `simulate` remains jittable.
    """

    def on_start(self) -> None: ...
    def on_outer_step(self, t, info: dict) -> None: ...
    def on_end(self) -> None: ...


class _SilentLogger:
    """No logging at all"""

    def on_start(self) -> None:
        pass

    def on_outer_step(self, t, info: dict) -> None:
        pass

    def on_end(self) -> None:
        pass


class _BeginEndLogger:
    """Log beginning and end and count elapsed time"""

    def __init__(self):
        self._start_time: float | None = None

    def on_start(self) -> None:
        print("Simulation started.")
        self._start_time = time.time()

    def on_outer_step(self, t, info: dict) -> None:
        pass

    def on_end(self) -> None:
        elapsed = time.time() - self._start_time
        print(f"Total elapsed time: {_format_long_duration(elapsed)}")
        print("Simulation complete.")


class _StepsLogger:
    """Log progress at each outer step"""

    def __init__(self, n_total: int):
        self.n_total = n_total
        self._i = 0
        self._start_time: float | None = None
        self._last_time: float | None = None

    def on_start(self) -> None:
        print("Simulation started.")

    def _host_callback(self, t, info: dict) -> None:
        current = time.time()
        perc = self._i / max(self.n_total - 1, 1) * 100
        extra = ""
        if "n_inner" in info:
            extra = f", inner steps: {int(info['n_inner'])}"
        if "dt_min" in info:
            extra += f", dt_min: {float(info['dt_min']):.3f}s"

        if self._last_time is None:
            self._start_time = current
            print(f"t={float(t):.0f} ({perc:.0f}%){extra}")
        else:
            duration = current - self._last_time
            eta = duration * (self.n_total - self._i)
            print(
                f"t={float(t):.0f} ({perc:.0f}%), this iter: {duration:.2f}s, ETA: {_format_long_duration(eta)}{extra}"
            )
        self._last_time = current
        self._i += 1

    def on_outer_step(self, t, info: dict) -> None:
        jax.debug.callback(self._host_callback, t, info)

    def on_end(self) -> None:
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            print(f"Total elapsed time: {_format_long_duration(elapsed)}")
        print("Simulation complete.")


def make_logger(level: LogLevel, n_outer: int) -> SimLogger:
    """Return a logger instance based on the specified log level."""
    if level == LogLevel.SILENT:
        return _SilentLogger()
    if level == LogLevel.BEGIN_END:
        return _BeginEndLogger()
    if level == LogLevel.STEPS:
        return _StepsLogger(n_total=n_outer)
    raise ValueError(f"Unknown log level: {level}")
