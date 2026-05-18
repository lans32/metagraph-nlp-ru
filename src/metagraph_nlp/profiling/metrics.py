"""Метрики стадий pipeline (CLAUDE.md §12.8)."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class StageMetrics:
    stage: str
    wall_seconds: float = 0.0
    peak_memory_kb: float = 0.0
    output_count: int = 0


@dataclass
class PipelineMetrics:
    stages: list[StageMetrics] = field(default_factory=list)
    total_wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_wall_seconds": round(self.total_wall_seconds, 4),
            "stages": [
                {
                    "stage": s.stage,
                    "wall_seconds": round(s.wall_seconds, 4),
                    "peak_memory_kb": round(s.peak_memory_kb, 2),
                    "output_count": s.output_count,
                }
                for s in self.stages
            ],
        }


@contextmanager
def measure_stage(
    name: str, metrics: PipelineMetrics
) -> Generator[StageMetrics, None, None]:
    sm = StageMetrics(stage=name)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield sm
    finally:
        sm.wall_seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sm.peak_memory_kb = peak / 1024
        metrics.stages.append(sm)
        metrics.total_wall_seconds += sm.wall_seconds
