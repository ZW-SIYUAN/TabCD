"""Enforce compilation deadlines and sampled process RSS limits."""

from __future__ import annotations

import time
from contextlib import contextmanager

from cdtab.errors import RelationMaterializationLimitError, ResourceMonitorError


class ResourceBudget:
    """Check compilation time and process RSS at bounded checkpoints."""

    def __init__(self, limits):
        self.limits = limits
        self.started = time.perf_counter()
        self.stage = "admission"
        self.phase_seconds: dict[str, float] = {}
        self.checkpoints = 0
        self.peak_observed_rss_bytes = 0
        try:
            import psutil

            self._process = psutil.Process()
        except Exception as error:
            raise ResourceMonitorError("psutil is required for process RSS accounting") from error
        self.check(force=True)
        self.initial_rss_bytes = self.peak_observed_rss_bytes

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def fail(self, message: str) -> None:
        error = RelationMaterializationLimitError(f"{self.stage}: {message}")
        error.stage = self.stage
        error.resource_diagnostics = self.snapshot()
        raise error

    def check(self, *, force: bool = False) -> None:
        self.checkpoints += 1
        if self.elapsed > self.limits.max_compile_seconds:
            self.fail("exceeded max_compile_seconds")
        if force or self.checkpoints % self.limits.rss_check_interval == 0:
            try:
                rss = int(self._process.memory_info().rss)
            except Exception as error:
                raise ResourceMonitorError(
                    "cannot read process RSS; compilation stopped"
                ) from error
            self.peak_observed_rss_bytes = max(self.peak_observed_rss_bytes, rss)
            if rss > self.limits.max_process_rss_bytes:
                self.fail("Exceeded max_process_rss_bytes")

    @contextmanager
    def phase(self, name: str):
        previous = self.stage
        self.stage = name
        started = time.perf_counter()
        try:
            self.check(force=True)
            yield
            self.check(force=True)
        except MemoryError:
            self.fail("Allocation failed")
        finally:
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )
            self.stage = previous

    def snapshot(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed,
            "phase_seconds": dict(self.phase_seconds),
            "initial_rss_bytes": getattr(self, "initial_rss_bytes", 0),
            "peak_observed_rss_bytes": self.peak_observed_rss_bytes,
            "resource_checkpoints": self.checkpoints,
        }
