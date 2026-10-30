"""Throughput counters and stage timers for scalability report (M1.7)."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ThroughputCounter:
    """Counts items processed and exposes throughput over wall-clock time."""
    name: str
    n: int = 0
    bytes_in: int = 0
    t_start: float = field(default_factory=time.monotonic)

    def add(self, n: int = 1, bytes_in: int = 0):
        self.n += n
        self.bytes_in += bytes_in

    @property
    def elapsed_s(self) -> float:
        return max(time.monotonic() - self.t_start, 1e-9)

    @property
    def rate(self) -> float:
        """Items per second."""
        return self.n / self.elapsed_s

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "elapsed_s": round(self.elapsed_s, 3),
            "rate_per_s": round(self.rate, 3),
            "bytes_in": self.bytes_in,
        }


@contextmanager
def stage_timer(stage: str, *, logger=None):
    """Context manager that logs wall-time of a stage.

    Usage:
        with stage_timer("parse_batch", logger=log):
            ...
    """
    t0 = time.monotonic()
    payload = {"stage": stage}
    try:
        yield payload
    except Exception as e:
        payload["error"] = type(e).__name__
        raise
    finally:
        payload["wall_s"] = round(time.monotonic() - t0, 3)
        if logger is not None:
            logger.info("stage_done", **payload)
