#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import platform
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULT_N = 4096
DEFAULT_D = 256
DEFAULT_TRANSFORM_VERSION = "0.0.0"
DEFAULT_TRANSFORMED_SEQ_DIVISOR = 4
DEFAULT_TRANSFORMED_DIM_DIVISOR = 4
INPUT_VALUE_MIN = -25.0
INPUT_VALUE_MAX = 25.0
DTYPE_SPECS: list[tuple[str, torch.dtype]] = [
    ("float16", torch.float16),
    ("float32", torch.float32),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


def load_thresholds(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(key): float(value) for key, value in data.items()}


def load_shapes(path: Path) -> list[dict[str, int]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        {
            "n": int(item.get("n", DEFAULT_N)),
            "d": int(item.get("d", DEFAULT_D)),
            "n_transformed": int(item["n_transformed"]) if "n_transformed" in item else (
                int(item["n_reduced"]) if "n_reduced" in item else 0
            ),
            "d_transformed": int(item["d_transformed"]) if "d_transformed" in item else (
                int(item["d_reduced"]) if "d_reduced" in item else 0
            ),
        }
        for item in data.get("default", [])
    ]


def choose_transformed_extent(original_extent: int, divisor: int) -> int:
    if original_extent <= 0 or divisor <= 0:
        return 0
    candidate = original_extent // divisor
    if candidate <= 0:
        return 1
    if original_extent % candidate == 0:
        return candidate
    for probe in range(candidate, 0, -1):
        if original_extent % probe == 0:
            return probe
    return 1


def default_transformed_shape(transform_version: str, n: int, d: int) -> tuple[int, int]:
    if transform_version != DEFAULT_TRANSFORM_VERSION:
        raise ValueError(f"no default transformed-shape policy registered for version: {transform_version}")
    return (
        choose_transformed_extent(n, DEFAULT_TRANSFORMED_SEQ_DIVISOR),
        choose_transformed_extent(d, DEFAULT_TRANSFORMED_DIM_DIVISOR),
    )


def resolve_case_shape(case: dict[str, int], transform_version: str) -> dict[str, int]:
    resolved = dict(case)
    if resolved["n_transformed"] == 0 or resolved["d_transformed"] == 0:
        resolved["n_transformed"], resolved["d_transformed"] = default_transformed_shape(
            transform_version,
            resolved["n"],
            resolved["d"],
        )
    return resolved


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


def generate_qkv(n: int, d: int, seed: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    def make_tensor() -> torch.Tensor:
        tensor = torch.empty((n, d), dtype=torch.float32)
        tensor.uniform_(INPUT_VALUE_MIN, INPUT_VALUE_MAX, generator=g)
        return tensor.to(device=device, dtype=dtype)

    q = make_tensor()
    k = make_tensor()
    v = make_tensor()
    return q, k, v


def transform_matrix(matrix: torch.Tensor, transformed_rows: int, transformed_cols: int) -> torch.Tensor:
    rows, cols = matrix.shape
    if rows % transformed_rows != 0 or cols % transformed_cols != 0:
        raise ValueError(f"shape {tuple(matrix.shape)} must be divisible by {(transformed_rows, transformed_cols)}")
    row_block = rows // transformed_rows
    col_block = cols // transformed_cols
    return matrix.view(transformed_rows, row_block, transformed_cols, col_block).mean(dim=(1, 3))


def reconstruct_matrix(matrix: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    transformed_rows, transformed_cols = matrix.shape
    if rows % transformed_rows != 0 or cols % transformed_cols != 0:
        raise ValueError(f"shape {(rows, cols)} must be divisible by {tuple(matrix.shape)}")
    row_block = rows // transformed_rows
    col_block = cols // transformed_cols
    return matrix.repeat_interleave(row_block, dim=0).repeat_interleave(col_block, dim=1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[1] ** -0.5
    scores = (q @ k.transpose(0, 1)) * scale
    probs = torch.softmax(scores, dim=1)
    return probs @ v


def synchronized_ms(fn) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def frobenius_relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denom = torch.linalg.norm(reference).item()
    if denom == 0.0:
        return 0.0
    return (torch.linalg.norm(reference - candidate) / denom).item()


def bounded_accuracy(relative_error: float) -> float:
    return max(0.0, 1.0 - relative_error)


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    avg_accuracy = sum(case["accuracy"] for case in cases) / max(len(cases), 1)
    avg_error = sum(case["relative_error"] for case in cases) / max(len(cases), 1)
    avg_exact_ms = sum(case["exact_latency_ms"] for case in cases) / max(len(cases), 1)
    avg_transformed_ms = sum(case["transformed_latency_ms"] for case in cases) / max(len(cases), 1)
    avg_exact_mem = sum(case["exact_memory_mb"] for case in cases) / max(len(cases), 1)
    avg_transformed_mem = sum(case["transformed_memory_mb"] for case in cases) / max(len(cases), 1)
    return {
        "average_accuracy": avg_accuracy,
        "average_relative_error": avg_error,
        "average_exact_latency_ms": avg_exact_ms,
        "average_transformed_latency_ms": avg_transformed_ms,
        "average_exact_memory_mb": avg_exact_mem,
        "average_transformed_memory_mb": avg_transformed_mem,
    }


def make_case_seeds(case_count: int, dtype_name: str, base_seed: int | None) -> list[int]:
    if base_seed is not None:
        dtype_offset = 0 if dtype_name == "float16" else 10_000
        return [base_seed + dtype_offset + idx for idx in range(case_count)]
    return [secrets.randbelow(2**31 - 1) + 1 for _ in range(case_count)]


def measure_case(case: dict[str, int], seed: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    n = case["n"]
    d = case["d"]
    n_transformed = case["n_transformed"]
    d_transformed = case["d_transformed"]
    q, k, v = generate_qkv(n, d, seed, device, dtype)

    torch.cuda.reset_peak_memory_stats(device)
    exact, exact_latency_ms = synchronized_ms(lambda: attention(q, k, v))
    exact_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    torch.cuda.reset_peak_memory_stats(device)

    def transformed():
        transformed_q = transform_matrix(q, n_transformed, d_transformed)
        transformed_k = transform_matrix(k, n_transformed, d_transformed)
        transformed_v = transform_matrix(v, n_transformed, d_transformed)
        transformed_out = attention(transformed_q, transformed_k, transformed_v)
        return reconstruct_matrix(transformed_out, n, d)

    approx, transformed_latency_ms = synchronized_ms(transformed)
    transformed_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    relative_error = frobenius_relative_error(exact, approx)
    accuracy = bounded_accuracy(relative_error)

    return {
        "n": n,
        "d": d,
        "n_transformed": n_transformed,
        "d_transformed": d_transformed,
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
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
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)
        return {"command": cmd, "status": "ok", "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except Exception as exc:
        return {"command": cmd, "status": "failed", "error": str(exc)}


def baseline_dtype_summary(baseline: dict[str, Any], dtype_name: str) -> dict[str, Any]:
    return baseline.get("dtype_summaries", {}).get(dtype_name, baseline.get("summary", {}))


def make_gpu_report(transform_version: str, base_seed: int | None) -> dict[str, Any]:
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
    thresholds = load_thresholds(root / "eval" / "thresholds.yaml")
    shapes = [resolve_case_shape(case, transform_version) for case in load_shapes(root / "bench" / "shapes.yaml")]

    baseline_path = root / "eval" / "baselines" / "0.0.0.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    dtype_cases: dict[str, list[dict[str, Any]]] = {}
    dtype_summaries: dict[str, dict[str, Any]] = {}
    all_cases: list[dict[str, Any]] = []

    for dtype_name, dtype in DTYPE_SPECS:
        seeds = make_case_seeds(len(shapes), dtype_name, base_seed)
        cases = [measure_case(case, seeds[idx], device, dtype) for idx, case in enumerate(shapes)]
        summary = summarize_cases(cases)
        baseline_summary = baseline_dtype_summary(baseline, dtype_name)
        accuracy_pass = summary["average_accuracy"] >= thresholds["accuracy_floor"]
        latency_limit = float(
            baseline_summary.get("average_transformed_latency_ms", summary["average_transformed_latency_ms"])
        ) * (1.0 + thresholds["max_latency_regression_pct"] / 100.0)
        memory_limit = float(
            baseline_summary.get("average_transformed_memory_mb", summary["average_transformed_memory_mb"])
        ) * (1.0 + thresholds["max_memory_regression_pct"] / 100.0)
        latency_pass = summary["average_transformed_latency_ms"] <= latency_limit
        memory_pass = summary["average_transformed_memory_mb"] <= memory_limit
        summary.update(
            {
                "accuracy_pass": accuracy_pass,
                "latency_pass": latency_pass,
                "memory_pass": memory_pass,
            }
        )
        dtype_cases[dtype_name] = cases
        dtype_summaries[dtype_name] = summary
        all_cases.extend(cases)

    overall_summary = summarize_cases(all_cases)
    overall_summary.update(
        {
            "accuracy_pass": all(summary["accuracy_pass"] for summary in dtype_summaries.values()),
            "latency_pass": all(summary["latency_pass"] for summary in dtype_summaries.values()),
            "memory_pass": all(summary["memory_pass"] for summary in dtype_summaries.values()),
        }
    )

    build = run_command(["bash", "scripts/build.sh"], root)
    tests = run_command(["bash", "scripts/test.sh"], root)
    bench = run_command(["bash", "bench/scripts/bench.sh"], root)

    verdict = (
        "pass"
        if overall_summary["accuracy_pass"]
        and overall_summary["latency_pass"]
        and overall_summary["memory_pass"]
        and build["status"] == "ok"
        and tests["status"] == "ok"
        and bench["status"] == "ok"
        else "fail"
    )

    return {
        "repo": "Cuda-Compute-OSS",
        "mode": "gpu",
        "transform_version": transform_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dtypes": [dtype_name for dtype_name, _ in DTYPE_SPECS],
        "seed_mode": "fixed" if base_seed is not None else "unpredictable",
        "base_seed": base_seed,
        "input_distribution": {
            "type": "uniform",
            "min": INPUT_VALUE_MIN,
            "max": INPUT_VALUE_MAX,
        },
        "gpu_environment": info,
        "thresholds": thresholds,
        "cases": all_cases,
        "dtype_cases": dtype_cases,
        "dtype_summaries": dtype_summaries,
        "summary": overall_summary,
        "commands": {"build": build, "tests": tests, "bench": bench},
        "verdict": verdict,
        "notes": [
            "This is the authoritative real eval path for CCO.",
            "Attention correctness, latency, and memory are measured on GPU after Q/K/V reduction and output reconstruction.",
            "Transform targets are flexible: any (n', d') that cleanly divides (n, d) is valid.",
            "Each eval run measures both float16 and float32.",
            "Q/K/V inputs are randomly generated from (n, d) with values in (-25, 25). Unless --seed is provided, eval uses unpredictable per-case seeds.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCO real GPU eval.")
    parser.add_argument("--transform-version", default="0.0.0")
    parser.add_argument("--report-path", default="eval/reports/gpu-latest.json")
    parser.add_argument("--markdown-path", default="eval/reports/gpu-latest.md")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed base seed for reproducible Q/K/V generation.")
    args = parser.parse_args()

    root = repo_root()
    report = make_gpu_report(args.transform_version, args.seed)
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
        f"- seed mode: `{report['seed_mode']}`",
        f"- input distribution: `uniform({report['input_distribution']['min']}, {report['input_distribution']['max']})`",
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
                "## Overall Summary",
                "",
                f"- average accuracy: `{report['summary']['average_accuracy']:.6f}`",
                f"- average relative error: `{report['summary']['average_relative_error']:.6f}`",
                f"- average transformed latency (ms): `{report['summary']['average_transformed_latency_ms']:.3f}`",
                f"- average transformed memory (MB): `{report['summary']['average_transformed_memory_mb']:.3f}`",
            ]
        )
    for dtype_name, summary in report.get("dtype_summaries", {}).items():
        lines.extend(
            [
                "",
                f"## {dtype_name}",
                "",
                f"- average accuracy: `{summary['average_accuracy']:.6f}`",
                f"- average relative error: `{summary['average_relative_error']:.6f}`",
                f"- average transformed latency (ms): `{summary['average_transformed_latency_ms']:.3f}`",
                f"- average transformed memory (MB): `{summary['average_transformed_memory_mb']:.3f}`",
                f"- verdict gates: `accuracy={summary['accuracy_pass']}` `latency={summary['latency_pass']}` `memory={summary['memory_pass']}`",
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
