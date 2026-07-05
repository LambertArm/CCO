#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


def summary_score(report: dict[str, Any]) -> float:
    summary = report.get("summary", {})
    if "total_score" in summary:
        return float(summary["total_score"])
    error = float(summary.get("average_relative_error", 1.0))
    latency = float(summary.get("average_transformed_latency_ms", 0.0))
    memory = float(summary.get("average_transformed_memory_mb", 0.0))
    if latency <= 0.0 or memory <= 0.0:
        return 0.0
    return (1.0 / (1.0 + error)) * (1.0 / latency) * (1.0 / memory)
