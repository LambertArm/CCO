#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_thresholds(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(key): float(value) for key, value in data.items()}


def load_shapes(path: Path) -> list[dict[str, int]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [{"n": int(item["n"]), "m": int(item["m"])} for item in data.get("default", [])]


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "missing"}
    props = torch.cuda.get_device_properties(0)
    return {
        "status": "ok",
        "name": props.name,
        "total_memory_mb": props.total_memory // (1024 * 1024),
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def generate_matrix_pair(n: int, seed: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    a = torch.randn((n, n), generator=g, dtype=dtype).to(device)
    b = torch.randn((n, n), generator=g, dtype=dtype).to(device)
    return a, b


def reduce_matrix(matrix: torch.Tensor, m: int) -> torch.Tensor:
    n = matrix.shape[0]
    if n % m != 0:
        raise ValueError(f"n={n} must be divisible by m={m}")
    block = n // m
    return matrix.view(m, block, m, block).mean(dim=(1, 3))


def reconstruct_matrix(matrix: torch.Tensor, n: int) -> torch.Tensor:
    m = matrix.shape[0]
    if n % m != 0:
        raise ValueError(f"n={n} must be divisible by m={m}")
    block = n // m
    return matrix.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)


def synchronized_ms(fn) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return value, elapsed_ms


def frobenius_relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denom = torch.linalg.norm(reference).item()
    if denom == 0.0:
        return 0.0
    return (torch.linalg.norm(reference - candidate) / denom).item()


def bounded_accuracy(relative_error: float) -> float:
    return max(0.0, 1.0 - relative_error)


def measure_case(case: dict[str, int], seed: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    n = case["n"]
    m = case["m"]
    a, b = generate_matrix_pair(n, seed, device, dtype)

    torch.cuda.reset_peak_memory_stats(device)
    exact, exact_latency_ms = synchronized_ms(lambda: a @ b)
    exact_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    torch.cuda.reset_peak_memory_stats(device)

    def transformed():
        reduced_a = reduce_matrix(a, m)
        reduced_b = reduce_matrix(b, m)
        reduced_c = reduced_a @ reduced_b
        return reconstruct_matrix(reduced_c, n)

    approx, transformed_latency_ms = synchronized_ms(transformed)
    transformed_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    relative_error = frobenius_relative_error(exact, approx)
    accuracy = bounded_accuracy(relative_error)

    return {
        "n": n,
        "m": m,
        "seed": seed,
        "dtype": str(dtype),
        "exact_latency_ms": exact_latency_ms,
        "transformed_latency_ms": transformed_latency_ms,
        "accuracy": accuracy,
        "relative_error": relative_error,
        "exact_memory_mb": exact_memory_mb,
        "transformed_memory_mb": transformed_memory_mb,
        "device": str(device),
    }


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return {"command": cmd, "status": "ok", "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except Exception as exc:
        return {"command": cmd, "status": "failed", "error": str(exc)}


def make_gpu_report(transform_version: str) -> dict[str, Any]:
    root = repo_root()
    info = gpu_info()
    if info["status"] != "ok":
        return {
            "repo": "Cuda-Compute-OSS",
            "mode": "gpu",
            "transform_version": transform_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(root),
            "gpu_environment": info,
            "verdict": "fail",
            "notes": ["No CUDA-capable PyTorch device detected. Real eval requires a GPU machine."],
        }

    device = torch.device("cuda:0")
    dtype = torch.float32
    thresholds = load_thresholds(root / "eval" / "thresholds.yaml")
    shapes = load_shapes(root / "bench" / "shapes.yaml")
    cases = [measure_case(case, 1000 + idx, device, dtype) for idx, case in enumerate(shapes)]

    avg_accuracy = sum(case["accuracy"] for case in cases) / max(len(cases), 1)
    avg_error = sum(case["relative_error"] for case in cases) / max(len(cases), 1)
    avg_exact_ms = sum(case["exact_latency_ms"] for case in cases) / max(len(cases), 1)
    avg_transformed_ms = sum(case["transformed_latency_ms"] for case in cases) / max(len(cases), 1)
    avg_exact_mem = sum(case["exact_memory_mb"] for case in cases) / max(len(cases), 1)
    avg_transformed_mem = sum(case["transformed_memory_mb"] for case in cases) / max(len(cases), 1)

    baseline_path = root / "eval" / "baselines" / "0.0.0.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    baseline_summary = baseline.get("summary", {})

    accuracy_pass = avg_accuracy >= thresholds["accuracy_floor"]
    latency_limit = float(baseline_summary.get("average_transformed_latency_ms", avg_transformed_ms)) * (
        1.0 + thresholds["max_latency_regression_pct"] / 100.0
    )
    memory_limit = float(baseline_summary.get("average_transformed_memory_mb", avg_transformed_mem)) * (
        1.0 + thresholds["max_memory_regression_pct"] / 100.0
    )
    latency_pass = avg_transformed_ms <= latency_limit
    memory_pass = avg_transformed_mem <= memory_limit

    build = run_command(["bash", "scripts/build.sh"], root)
    tests = run_command(["bash", "scripts/test.sh"], root)
    bench = run_command(["bash", "bench/scripts/bench.sh"], root)

    build_ok = build["status"] == "ok"
    tests_ok = tests["status"] == "ok"
    bench_ok = bench["status"] == "ok"
    verdict = "pass" if accuracy_pass and latency_pass and memory_pass and build_ok and tests_ok and bench_ok else "fail"

    return {
        "repo": "Cuda-Compute-OSS",
        "mode": "gpu",
        "transform_version": transform_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dtype": "float32",
        "gpu_environment": info,
        "thresholds": thresholds,
        "cases": cases,
        "summary": {
            "average_accuracy": avg_accuracy,
            "average_relative_error": avg_error,
            "average_exact_latency_ms": avg_exact_ms,
            "average_transformed_latency_ms": avg_transformed_ms,
            "average_exact_memory_mb": avg_exact_mem,
            "average_transformed_memory_mb": avg_transformed_mem,
            "accuracy_pass": accuracy_pass,
            "latency_pass": latency_pass,
            "memory_pass": memory_pass,
        },
        "commands": {
            "build": build,
            "tests": tests,
            "bench": bench,
        },
        "verdict": verdict,
        "notes": [
            "This is the authoritative real eval path for CCO.",
            "Matrix computation, accuracy, latency, and memory are measured on GPU.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCO real GPU eval.")
    parser.add_argument("--transform-version", default="0.0.0")
    parser.add_argument("--report-path", default="eval/reports/gpu-latest.json")
    parser.add_argument("--markdown-path", default="eval/reports/gpu-latest.md")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    report = make_gpu_report(args.transform_version)
    report_path = root / args.report_path
    markdown_path = root / args.markdown_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CCO GPU Eval Report",
        "",
        f"- transform version: `{report['transform_version']}`",
        f"- git commit: `{report['git_commit']}`",
        f"- verdict: `{report['verdict']}`",
        "",
        "## GPU Environment",
        "",
        f"- status: `{report['gpu_environment']['status']}`",
    ]
    if report["gpu_environment"].get("status") == "ok":
        lines.append(f"- gpu: `{report['gpu_environment']['name']}`")
        lines.append(f"- cuda: `{report['gpu_environment']['cuda_runtime']}`")
    if "summary" in report:
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- average accuracy: `{report['summary']['average_accuracy']:.6f}`",
                f"- average relative error: `{report['summary']['average_relative_error']:.6f}`",
                f"- average transformed latency (ms): `{report['summary']['average_transformed_latency_ms']:.3f}`",
                f"- average transformed memory (MB): `{report['summary']['average_transformed_memory_mb']:.3f}`",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.write_baseline and report["verdict"] == "pass":
        baseline_path = root / "eval" / "baselines" / f"{args.transform_version}.json"
        baseline_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("CCO gpu eval")
    print(f"json report: {report_path}")
    print(f"markdown report: {markdown_path}")
    if args.write_baseline and report["verdict"] == "pass":
        print(f"baseline updated: {root / 'eval' / 'baselines' / f'{args.transform_version}.json'}")
    print(f"verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
