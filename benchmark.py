#!/usr/bin/env python3
"""
benchmark.py -- CCO benchmark harness (LOCKED -- byte-verified against manifest.json; the
miner NEVER modifies this file).

Handles:
  1. GPU hardware detection and roofline modelling
  2. Correctness verification (5 stages)
  3. Performance benchmarking (Triton do_bench)
  4. Structured, greppable output + the bound score blob (--blob)

Usage:
  uv run benchmark.py                        # benchmark kernel.py using its KERNEL_TYPE
  uv run benchmark.py --kernel matmul        # force kernel type
  uv run benchmark.py --quick                # skip stages 3-5, bench only large size
  uv run benchmark.py --score                # emit the competition latency sample
  uv run benchmark.py --blob                 # emit the bound score blob the canonical rerun verifies
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

from kernel_configs import KERNEL_CONFIGS


def _emit_json(metrics: dict, stream=None) -> None:
    print(json.dumps(metrics, sort_keys=True), file=stream or sys.stdout)


def _build_json_metrics(
    kernel_type: str,
    correctness_results: dict,
    primary: dict | None,
    perf_results: dict,
    gpu: GPUSpec,
    peak_vram_mb: float,
    t_elapsed: float,
) -> dict:
    metrics = {
        "kernel_type": kernel_type,
        "gpu_name": gpu.name,
        "gpu_sm_count": gpu.sm_count,
        "gpu_memory_gb": gpu.memory_gb,
        "gpu_peak_tflops_fp16": gpu.peak_tflops_fp16,
        "gpu_peak_tflops_bf16": gpu.peak_tflops_bf16,
        "gpu_peak_tflops_fp32": gpu.peak_tflops_fp32,
        "gpu_peak_bandwidth_gb_s": gpu.peak_bandwidth_gb_s,
        "gpu_l2_cache_mb": gpu.l2_cache_mb,
        "gpu_compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
        "correctness": correctness_results["correctness"],
        "smoke_test": correctness_results.get("smoke_test", "N/A"),
        "shape_sweep": correctness_results.get("shape_sweep", "N/A"),
        "numerical_stability": correctness_results.get("numerical_stability", "N/A"),
        "determinism": correctness_results.get("determinism", "N/A"),
        "edge_cases": correctness_results.get("edge_cases", "N/A"),
        "throughput_tflops": 0.0,
        "bandwidth_gb_s": 0.0,
        "pct_peak_compute": 0.0,
        "pct_peak_bandwidth": 0.0,
        "bottleneck": "",
        "speedup_vs_pytorch": 0.0,
        "peak_vram_mb": peak_vram_mb,
        "bench_time_seconds": t_elapsed,
        "performance_all": perf_results.get("all", []),
    }

    if primary:
        metrics.update({
            "latency_us": primary["kernel_latency_us"],
            "latency_ms": primary["kernel_latency_us"] / 1000.0,
            "pytorch_latency_us": primary["pytorch_latency_us"],
            "pytorch_latency_ms": primary["pytorch_latency_us"] / 1000.0,
            "kernel_latency_us": primary["kernel_latency_us"],
            "kernel_latency_ms": primary["kernel_latency_us"] / 1000.0,
            "throughput_tflops": primary["throughput_tflops"],
            "bandwidth_gb_s": primary["bandwidth_gb_s"],
            "pct_peak_compute": primary["pct_peak_compute"],
            "pct_peak_bandwidth": primary["pct_peak_bandwidth"],
            "arithmetic_intensity": primary["arithmetic_intensity"],
            "ridge_point": primary["ridge_point"],
            "bottleneck": primary["bottleneck"],
            "flops": primary["flops"],
            "bytes": primary["bytes"],
            "speedup_vs_pytorch": primary["speedup_vs_pytorch"],
            "pytorch_tflops": primary["ref_throughput_tflops"],
            "kernel_tflops": primary["throughput_tflops"],
            "primary_label": primary["label"],
            "primary_size": primary["size"],
            "dtype": primary["dtype"],
        })
        metrics.update({k: v for k, v in primary.items() if k.startswith("ncu_")})
    else:
        metrics.update({
            "latency_us": 0.0,
            "latency_ms": 0.0,
            "pytorch_latency_us": 0.0,
            "pytorch_latency_ms": 0.0,
            "kernel_latency_us": 0.0,
            "kernel_latency_ms": 0.0,
        })

    return metrics


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class BenchTimeoutError(Exception):
    pass


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._use_signal = hasattr(signal, "SIGALRM")

    def _handler(self, signum, frame):
        raise BenchTimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self._use_signal:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        else:
            import threading
            self._timer = threading.Timer(self.seconds, self._timeout_thread)
            self._timer.daemon = True
            self._timed_out = False
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._use_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
            return False
        self._timer.cancel()
        if self._timed_out and exc_type is KeyboardInterrupt:
            # interrupt_main() surfaces as KeyboardInterrupt in the main thread; convert it so
            # callers catching BenchTimeoutError see a stage timeout instead of a full abort.
            raise BenchTimeoutError(f"Timed out after {self.seconds}s") from None
        return False

    def _timeout_thread(self):
        self._timed_out = True
        import _thread
        _thread.interrupt_main()


# =========================================================================
# 1. GPU HARDWARE DETECTION
# =========================================================================

@dataclass
class GPUSpec:
    name: str = "Unknown"
    sm_count: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)


_KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
    "H800":       (989.5,  3352.0, 50.0),
    "H100 SXM":   (989.5,  3352.0, 50.0),
    "H100 PCIe":  (756.0,  2039.0, 50.0),
    "H100":       (756.0,  2039.0, 50.0),
    "H200":       (989.5,  4800.0, 50.0),
    "A100-SXM":   (312.0,  2039.0, 40.0),
    "A100-PCIE":  (312.0,  1935.0, 40.0),
    "A100":       (312.0,  2039.0, 40.0),
    "L40S":       (362.05, 864.0,  48.0),
    "L4":         (121.0,  300.0,  48.0),
    "A10":        (125.0,  600.0,  6.0),
    "4090":       (330.0,  1008.0, 72.0),
    "4080":       (305.0,  716.8,  64.0),
    "3090":       (142.0,  936.2,  6.0),
    "3080":       (119.5,  760.3,  5.0),
    # Blackwell GeForce (RTX 50xx). fp16/L2 here are approximate and feed only the %-peak
    # DISPLAY, never the scored speedup ratio; the 5070 Ti's 896 GB/s bandwidth is exact.
    # The canonical competition SKU's real peaks get pinned in runtime/gpu.config (Step 16).
    "5070 Ti":    (177.0,  896.0,  48.0),
    "B200":       (2250.0, 8000.0, 64.0),
    "B100":       (1800.0, 8000.0, 64.0),
}


def detect_gpu() -> GPUSpec:
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected, using dummy spec")
        return GPUSpec()

    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    memory_gb = round(props.total_memory / (1024 ** 3), 1)
    cc = (props.major, props.minor)

    matched = None
    for fragment, specs in _KNOWN_GPUS.items():
        if fragment in name:
            matched = specs
            break

    if matched is not None:
        peak_fp16, peak_bw, l2 = matched
    else:
        ops_per_clock_per_sm = 256 if cc[0] >= 8 else 128
        # torch >= 2.8 may not expose props.clock_rate; degrade gracefully instead of crashing.
        clock_khz = getattr(props, "clock_rate", 0) or 0
        clock_ghz = (clock_khz / 1e6) if clock_khz else 1.5  # GHz; fall back to a typical boost clock
        peak_fp16 = sm_count * ops_per_clock_per_sm * clock_ghz * 2 / 1e3
        peak_bw = max(clock_ghz * 256 / 8 * 2, 500.0)
        l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, "L2_cache_size") else 0.0

    peak_bf16 = peak_fp16
    peak_fp32 = peak_fp16 / 2.0

    return GPUSpec(
        name=name,
        sm_count=sm_count,
        memory_gb=memory_gb,
        peak_tflops_fp16=peak_fp16,
        peak_tflops_bf16=peak_bf16,
        peak_tflops_fp32=peak_fp32,
        peak_bandwidth_gb_s=peak_bw,
        l2_cache_mb=l2,
        compute_capability=cc,
    )


# =========================================================================
# 2. CORRECTNESS TESTING (5 stages)
# =========================================================================

def _compare(output: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    if output.shape != expected.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {output.shape} vs {expected.shape}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }

    out_f = output.float()
    exp_f = expected.float()
    abs_diff = (out_f - exp_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    within = (abs_diff <= atol + rtol * exp_f.abs()).float().mean().item() * 100.0
    match = torch.allclose(out_f, exp_f, atol=atol, rtol=rtol)

    return {
        "match": match,
        "reason": "" if match else f"max_abs_error={max_abs:.6e} exceeds tol(atol={atol}, rtol={rtol})",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "pct_within_tol": within,
    }


def _compare_multi(output, expected, atol: float, rtol: float) -> dict:
    """Compare multi-output kernels (e.g. weight_quant returns tuple)."""
    if not isinstance(output, (tuple, list)):
        output = (output,)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)

    if len(output) != len(expected):
        return {
            "match": False,
            "reason": f"output count mismatch: {len(output)} vs {len(expected)}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }

    worst_error = 0.0
    for i, (o, e) in enumerate(zip(output, expected)):
        r = _compare(o, e, atol, rtol)
        if not r["match"]:
            r["reason"] = f"output[{i}]: {r['reason']}"
            return r
        worst_error = max(worst_error, r["max_abs_error"])

    return {
        "match": True,
        "reason": "",
        "max_abs_error": worst_error,
        "mean_abs_error": 0.0,
        "pct_within_tol": 100.0,
    }


def _has_nan_inf(t) -> bool:
    if isinstance(t, (tuple, list)):
        return any(_has_nan_inf(x) for x in t)
    try:
        return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())
    except (RuntimeError, NotImplementedError):
        return bool(torch.isnan(t.float()).any().item() or torch.isinf(t.float()).any().item())


def _do_compare(output, expected, atol, rtol, multi_output):
    if multi_output:
        return _compare_multi(output, expected, atol, rtol)
    return _compare(output, expected, atol, rtol)


def run_correctness(kernel_fn: Callable, config: dict, quick: bool = False, seed: int = 42) -> dict:
    device = "cuda"
    multi_output = config.get("multi_output", False)
    results = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "numerical_stability": "SKIP",
        "determinism": "SKIP",
        "edge_cases": "SKIP",
        "correctness": "FAIL",
    }
    details = []
    all_pass = True

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    tols = config["tolerances"]

    # ------------------------------------------------------------------
    # Stage 1: SMOKE TEST
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Smoke Test ---")
    try:
        _, tiny_size = sizes[0]
        dtype0 = dtypes[0]
        inputs = gen_fn(tiny_size, dtype0, device, seed=seed)
        expected = ref_fn(inputs)
        with _Timeout(30):
            output = kernel_fn(**inputs)

        if _has_nan_inf(output):
            results["smoke_test"] = "FAIL"
            details.append("  smoke: NaN/Inf in output")
            all_pass = False
            print("  FAIL: NaN/Inf in output")
        else:
            tol = tols.get(dtype0, {"atol": 1e-2, "rtol": 1e-2})
            cmp = _do_compare(output, expected, **tol, multi_output=multi_output)
            if cmp["match"]:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={cmp['max_abs_error']:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                details.append(f"  smoke: {cmp['reason']}")
                all_pass = False
                print(f"  FAIL: {cmp['reason']}")
    except BenchTimeoutError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: TIMEOUT")
        all_pass = False
        print("  FAIL: TIMEOUT")
    except torch.cuda.OutOfMemoryError:
        results["smoke_test"] = "FAIL"
        details.append("  smoke: OOM")
        all_pass = False
        print("  FAIL: OOM on tiny input")
    except Exception as e:
        results["smoke_test"] = "FAIL"
        details.append(f"  smoke: CRASH ({type(e).__name__}: {e})")
        all_pass = False
        print(f"  FAIL: CRASH ({type(e).__name__}: {e})")

    if results["smoke_test"] == "FAIL":
        results["correctness"] = "FAIL"
        results["details"] = details
        print("\ncorrectness: FAIL (smoke test failed, aborting remaining stages)")
        return results

    # ------------------------------------------------------------------
    # Stage 2: SHAPE SWEEP
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Shape Sweep ---")
    sweep_pass = True
    sweep_count = 0
    sweep_fail_count = 0
    sweep_run_count = 0
    worst_error = 0.0
    worst_case = ""

    for label, sz in sizes:
        for dtype in dtypes:
            sweep_count += 1
            try:
                inputs = gen_fn(sz, dtype, device, seed=seed)
                expected = ref_fn(inputs)
                with _Timeout(30):
                    output = kernel_fn(**inputs)

                sweep_run_count += 1

                if _has_nan_inf(output):
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: NaN/Inf")
                    print(f"  FAIL: {label} {dtype} -> NaN/Inf")
                    continue

                tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                cmp = _do_compare(output, expected, **tol, multi_output=multi_output)

                if cmp["max_abs_error"] > worst_error:
                    worst_error = cmp["max_abs_error"]
                    worst_case = f"{label}/{dtype}"

                if not cmp["match"]:
                    sweep_pass = False
                    sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: {cmp['reason']}")
                    print(f"  FAIL: {label} {dtype} -> {cmp['reason']}")
                else:
                    print(
                        f"  PASS: {label} {dtype} "
                        f"(max_err={cmp['max_abs_error']:.2e}, "
                        f"within_tol={cmp['pct_within_tol']:.1f}%)"
                    )

            except torch.cuda.OutOfMemoryError:
                print(f"  SKIP: {label} {dtype} -> OOM")
                torch.cuda.empty_cache()
                continue
            except BenchTimeoutError:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: TIMEOUT")
                print(f"  FAIL: {label} {dtype} -> TIMEOUT")
            except Exception as e:
                sweep_pass = False
                sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: {type(e).__name__}: {e}")
                print(f"  FAIL: {label} {dtype} -> {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()

    if sweep_run_count == 0:
        results["shape_sweep"] = f"FAIL (0/{sweep_count} configs ran — all skipped due to OOM)"
        all_pass = False
        print(f"  shape_sweep: FAIL (0/{sweep_count} configs ran — all skipped due to OOM)")
    elif sweep_pass:
        results["shape_sweep"] = (
            f"PASS ({sweep_run_count}/{sweep_count} configs, "
            f"worst_err={worst_error:.2e} at {worst_case})"
        )
        print(f"  shape_sweep: PASS ({sweep_run_count}/{sweep_count} configs, worst_err={worst_error:.2e})")
    else:
        results["shape_sweep"] = f"FAIL ({sweep_fail_count}/{sweep_count} failed)"
        all_pass = False
        print(f"  shape_sweep: FAIL ({sweep_fail_count}/{sweep_count} failed)")

    # ------------------------------------------------------------------
    # Stages 3-5: Skip in --quick mode
    # ------------------------------------------------------------------
    if quick:
        results["numerical_stability"] = "SKIP (quick mode)"
        results["determinism"] = "SKIP (quick mode)"
        results["edge_cases"] = "SKIP (quick mode)"
        results["correctness"] = "PASS" if all_pass else "FAIL"
        results["details"] = details
        print(f"\ncorrectness: {results['correctness']} (quick mode: stages 3-5 skipped)")
        return results

    # ------------------------------------------------------------------
    # Stage 3: NUMERICAL STABILITY
    # ------------------------------------------------------------------
    print("\n--- Stage 3: Numerical Stability ---")
    stability_pass = True
    stab_size = None
    for label, sz in sizes:
        if label == "small":
            stab_size = sz
            break
    if stab_size is None:
        stab_size = sizes[min(1, len(sizes) - 1)][1]
    stab_dtype = dtypes[0]

    adversarial_cases = [
        ("near_max", lambda t: t * 60000.0 if t.dtype == torch.float16 else t * 1e30),
        ("near_zero", lambda t: t * 1e-6),
        ("mixed_scale", lambda t: t * torch.where(
            torch.rand_like(t.float()).to(t.dtype) > 0.5,
            torch.tensor(1e3, device=t.device, dtype=t.dtype),
            torch.tensor(1e-3, device=t.device, dtype=t.dtype),
        )),
        ("all_zeros", lambda t: torch.zeros_like(t)),
        ("all_same", lambda t: torch.ones_like(t) * 0.5),
    ]

    for case_name, transform_fn in adversarial_cases:
        try:
            inputs = gen_fn(stab_size, stab_dtype, device, seed=seed)
            transformed = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    transformed[k] = transform_fn(v)
                else:
                    transformed[k] = v

            expected = ref_fn(transformed)
            with _Timeout(30):
                output = kernel_fn(**transformed)

            if _has_nan_inf(output) and not _has_nan_inf(expected):
                stability_pass = False
                details.append(f"  stability {case_name}: NaN/Inf (reference is clean)")
                print(f"  FAIL: {case_name} -> NaN/Inf (reference is clean)")
            elif _has_nan_inf(output) and _has_nan_inf(expected):
                print(f"  PASS: {case_name} -> both have NaN/Inf (expected overflow)")
            else:
                tol = tols.get(stab_dtype, {"atol": 1e-2, "rtol": 1e-2})
                relaxed_atol = tol["atol"] * 10
                relaxed_rtol = tol["rtol"] * 10
                cmp = _do_compare(output, expected, atol=relaxed_atol, rtol=relaxed_rtol, multi_output=multi_output)
                if cmp["match"]:
                    print(f"  PASS: {case_name} (max_err={cmp['max_abs_error']:.2e})")
                else:
                    stability_pass = False
                    details.append(f"  stability {case_name}: {cmp['reason']}")
                    print(f"  FAIL: {case_name} -> {cmp['reason']}")

        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: {case_name} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            stability_pass = False
            details.append(f"  stability {case_name}: TIMEOUT")
            print(f"  FAIL: {case_name} -> TIMEOUT")
        except Exception as e:
            stability_pass = False
            details.append(f"  stability {case_name}: {type(e).__name__}: {e}")
            print(f"  FAIL: {case_name} -> {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()

    results["numerical_stability"] = "PASS" if stability_pass else "FAIL"
    if not stability_pass:
        all_pass = False
    print(f"  numerical_stability: {results['numerical_stability']}")

    # ------------------------------------------------------------------
    # Stage 4: DETERMINISM
    # ------------------------------------------------------------------
    print("\n--- Stage 4: Determinism ---")
    determinism_pass = True
    try:
        det_size = stab_size
        det_dtype = dtypes[0]

        def _flatten(x):
            if isinstance(x, (tuple, list)):
                return torch.cat([t.flatten().float() for t in x])
            return x.flatten().float()

        outputs = []
        for _ in range(3):
            inputs_i = gen_fn(det_size, det_dtype, device, seed=seed)
            with _Timeout(30):
                out_i = kernel_fn(**inputs_i)
            outputs.append(_flatten(out_i))

        # Within-tolerance reproducibility, NOT bitwise. A correct kernel that uses atomics /
        # split-K / nondeterministic reductions varies by ~fp rounding run-to-run (far below the
        # correctness tolerance); bitwise-equality would wrongly reject those legitimately-fast
        # kernels. A gross race (output differing beyond tolerance) is still caught.
        det_tol = config["tolerances"].get(det_dtype, {"atol": 1e-2, "rtol": 1e-2})
        for i in range(1, 3):
            if not torch.allclose(outputs[0], outputs[i], atol=det_tol["atol"], rtol=det_tol["rtol"]):
                determinism_pass = False
                diff = (outputs[0] - outputs[i]).abs()
                details.append(f"  determinism: run 0 vs run {i} differ beyond tol (max_diff={diff.max().item():.6e})")
                print(f"  FAIL: run 0 vs run {i} differ beyond tol (max_diff={diff.max().item():.6e})")

        if determinism_pass:
            print("  PASS: 3 runs reproducible within tolerance")
        results["determinism"] = "PASS" if determinism_pass else "FAIL"
    except Exception as e:
        results["determinism"] = f"FAIL ({type(e).__name__})"
        all_pass = False
        details.append(f"  determinism: {type(e).__name__}: {e}")
        print(f"  FAIL: {type(e).__name__}: {e}")
    finally:
        torch.cuda.empty_cache()

    if not determinism_pass:
        all_pass = False

    # ------------------------------------------------------------------
    # Stage 5: EDGE CASES
    # ------------------------------------------------------------------
    print("\n--- Stage 5: Edge Cases ---")
    edge_pass = True
    edge_sizes = config.get("edge_sizes", [])
    if not edge_sizes:
        results["edge_cases"] = "SKIP (no edge sizes defined)"
        print("  SKIP: no edge sizes defined")
    else:
        for label, sz in edge_sizes:
            for dtype in dtypes[:1]:
                try:
                    inputs = gen_fn(sz, dtype, device, seed=seed)
                    expected = ref_fn(inputs)
                    with _Timeout(30):
                        output = kernel_fn(**inputs)

                    if _has_nan_inf(output) and not _has_nan_inf(expected):
                        edge_pass = False
                        details.append(f"  edge {label}: NaN/Inf")
                        print(f"  FAIL: {label} -> NaN/Inf")
                    else:
                        tol = tols.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                        cmp = _do_compare(output, expected, **tol, multi_output=multi_output)
                        if cmp["match"]:
                            print(f"  PASS: {label} (max_err={cmp['max_abs_error']:.2e})")
                        else:
                            edge_pass = False
                            details.append(f"  edge {label}: {cmp['reason']}")
                            print(f"  FAIL: {label} -> {cmp['reason']}")

                except torch.cuda.OutOfMemoryError:
                    print(f"  SKIP: {label} -> OOM")
                    torch.cuda.empty_cache()
                except BenchTimeoutError:
                    edge_pass = False
                    details.append(f"  edge {label}: TIMEOUT")
                    print(f"  FAIL: {label} -> TIMEOUT")
                except Exception as e:
                    edge_pass = False
                    details.append(f"  edge {label}: {type(e).__name__}: {e}")
                    print(f"  FAIL: {label} -> {type(e).__name__}: {e}")
                finally:
                    torch.cuda.empty_cache()

        results["edge_cases"] = "PASS" if edge_pass else "FAIL"
        if not edge_pass:
            all_pass = False
        print(f"  edge_cases: {results['edge_cases']}")

    results["correctness"] = "PASS" if all_pass else "FAIL"
    results["details"] = details
    print(f"\ncorrectness: {results['correctness']}")
    return results


# =========================================================================
# 3. PERFORMANCE BENCHMARKING
# =========================================================================

def _peak_tflops_for_dtype(gpu: GPUSpec, dtype: torch.dtype) -> float:
    """Return the GPU's peak TFLOPS for the given dtype."""
    if dtype in (torch.float32, torch.float64):
        return gpu.peak_tflops_fp32
    if dtype == torch.bfloat16:
        return gpu.peak_tflops_bf16
    return gpu.peak_tflops_fp16


def _do_bench(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark a function and return median time in milliseconds."""
    try:
        from triton.testing import do_bench
        ms = do_bench(fn, warmup=warmup, rep=rep)
        return ms
    except ImportError:
        pass

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def run_performance(kernel_fn: Callable, config: dict, gpu: GPUSpec,
                    sizes_filter: str = "all", seed: int = 42) -> dict:
    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    bytes_fn = config["bytes_fn"]
    dtypes = config["test_dtypes"]

    sizes = config["test_sizes"]
    bench_sizes = []
    if sizes_filter == "all":
        bench_sizes = sizes
    else:
        for label, sz in sizes:
            if label == sizes_filter:
                bench_sizes = [(label, sz)]
                break
        if not bench_sizes:
            for label, sz in sizes:
                if label == "large":
                    bench_sizes = [(label, sz)]
                    break
            if not bench_sizes:
                bench_sizes = [sizes[-1]]

    primary_label = None
    primary_size = None
    for label, sz in sizes:
        if label == "large":
            primary_label = label
            primary_size = sz
            break
    if primary_size is None:
        primary_label, primary_size = sizes[-1]

    dtype = dtypes[0]
    all_results = []
    primary_result = None

    for label, sz in bench_sizes:
        print(f"\n  Benchmarking: {label} ...")
        try:
            flops = flops_fn(sz)
            nbytes = bytes_fn(sz, dtype)

            inputs = gen_fn(sz, dtype, device, seed=seed)

            _k_inputs = inputs
            with _Timeout(30):
                kernel_ms = _do_bench(lambda _i=_k_inputs: kernel_fn(**_i), warmup=25, rep=100)

            with _Timeout(30):
                ref_ms = _do_bench(lambda _i=_k_inputs: ref_fn(_i), warmup=25, rep=100)

            kernel_us = kernel_ms * 1000.0
            ref_us = ref_ms * 1000.0
            throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
            bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
            ref_throughput_tflops = flops / (ref_ms / 1000.0) / 1e12 if ref_ms > 0 else 0.0

            peak_tflops = _peak_tflops_for_dtype(gpu, dtype)
            arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
            ridge_point = (peak_tflops * 1e12) / (gpu.peak_bandwidth_gb_s * 1e9) if gpu.peak_bandwidth_gb_s > 0 else 0.0

            if arithmetic_intensity < ridge_point:
                bottleneck = "memory_bound"
                pct_peak_bandwidth = (
                    (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0)
                    if gpu.peak_bandwidth_gb_s > 0
                    else 0.0
                )
                pct_peak_compute = (throughput_tflops / peak_tflops * 100.0) if peak_tflops > 0 else 0.0
            else:
                bottleneck = "compute_bound"
                pct_peak_compute = (throughput_tflops / peak_tflops * 100.0) if peak_tflops > 0 else 0.0
                pct_peak_bandwidth = (
                    (bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0)
                    if gpu.peak_bandwidth_gb_s > 0
                    else 0.0
                )

            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0

            entry = {
                "label": label,
                "size": sz,
                "dtype": str(dtype),
                "flops": flops,
                "bytes": nbytes,
                "kernel_latency_us": kernel_us,
                "pytorch_latency_us": ref_us,
                "throughput_tflops": throughput_tflops,
                "bandwidth_gb_s": bandwidth_gb_s,
                "ref_throughput_tflops": ref_throughput_tflops,
                "pct_peak_compute": pct_peak_compute,
                "pct_peak_bandwidth": pct_peak_bandwidth,
                "arithmetic_intensity": arithmetic_intensity,
                "ridge_point": ridge_point,
                "bottleneck": bottleneck,
                "speedup_vs_pytorch": speedup,
            }
            all_results.append(entry)

            if label == primary_label:
                primary_result = entry

            print(f"    kernel: {kernel_us:.2f} us | pytorch: {ref_us:.2f} us | "
                  f"speedup: {speedup:.3f}x | {throughput_tflops:.3f} TFLOPS | "
                  f"{pct_peak_compute:.1f}% peak")

        except torch.cuda.OutOfMemoryError:
            print(f"    SKIP: {label} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            print(f"    SKIP: {label} -> TIMEOUT")
        except Exception as e:
            print(f"    ERROR: {label} -> {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    if primary_result is None and all_results:
        primary_result = all_results[-1]

    return {
        "primary": primary_result,
        "all": all_results,
    }


# =========================================================================
# 4. PROFILER (optional)
# =========================================================================

def _tensor_storage_ptrs(x) -> set:
    """Underlying data pointers of a tensor or tuple/list of tensors (for the anti-alias guard)."""
    ptrs = set()
    items = x if isinstance(x, (tuple, list)) else [x]
    for t in items:
        if torch.is_tensor(t):
            ptrs.add(t.data_ptr())
    return ptrs


def run_scored_sample(kernel_fn: Callable, config: dict, seed: int = 42,
                      n_blocks: int = 30, warmup: int = 25, rep: int = 100,
                      n_buffers: int = 4) -> dict:
    """Competition scoring measurement (Step 7).

    Emits a SAMPLE of per-call latencies on the primary (scored) size + dtype for the
    challenger-vs-champion significance test (cco/significance.py). Anti-cheat properties baked in:
      * input buffers ROTATE across reps (distinct seeds -> distinct values & storage), so a kernel
        can't win via warm-L2 residency or memoize-by-pointer;
      * CORRECTNESS is re-verified on the scored size on 2 distinct buffers (a kernel that memoizes
        buffer-0's answer fails on buffer-1) -> closes "fast garbage at the scored size";
      * the output must not alias an input (anti-elision guard);
      * the validation calls run under the runtime no-delegation trap (cco/dispatch_trap.py) — a
        kernel that delegates at runtime is disqualified (delegation != None) and never timed.
    The latencies are n_blocks block-means (mean over `rep` rotated calls), each a fresh timed
    window so the sample captures run-to-run / thermal variation for the nonparametric test.
    """
    import statistics

    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    multi = config.get("multi_output", False)
    dtype = config["test_dtypes"][0]
    tols = config["tolerances"].get(dtype, {"atol": 1e-2, "rtol": 1e-2})

    sizes = config["test_sizes"]
    size_label, size = None, None
    for label, sz in sizes:
        if label == "large":
            size_label, size = label, sz
            break
    if size is None:
        size_label, size = sizes[-1]

    # rotating buffer pool: distinct seeds -> distinct values & storage addresses
    buffers = [gen_fn(size, dtype, device, seed=seed + i) for i in range(n_buffers)]

    # fused correctness on the scored size, on up to 2 distinct buffers, + anti-alias guard.
    # The validation calls run under the runtime no-delegation trap (cco/dispatch_trap.py) —
    # the trap's per-op interception overhead is why it wraps these and NOT the timed reps.
    from cco.dispatch_trap import DelegationError, run_guarded

    correct, worst_err, aliased, delegation = True, 0.0, False, None
    for i in range(min(2, n_buffers)):
        inp = buffers[i]
        try:
            out = run_guarded(kernel_fn, inp)
        except DelegationError as e:
            delegation = str(e)
            correct = False
            break
        cmp = _do_compare(out, ref_fn(inp), tols["atol"], tols["rtol"], multi)
        correct = correct and bool(cmp["match"])
        worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
        if i == 0:
            in_ptrs = set()
            for t in inp.values():
                in_ptrs |= _tensor_storage_ptrs(t)
            aliased = bool(_tensor_storage_ptrs(out) & in_ptrs)
    correct = correct and not aliased

    # timing: n_blocks block-means, rotating buffers, fp32 CUDA events.
    # A kernel caught delegating is disqualified — don't execute it further; emit an empty sample.
    latencies_us = []
    if delegation is None:
        for _ in range(warmup):
            kernel_fn(**buffers[0])
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _blk in range(n_blocks):
            start.record()
            for r in range(rep):
                kernel_fn(**buffers[r % n_buffers])
            end.record()
            torch.cuda.synchronize()
            latencies_us.append(start.elapsed_time(end) * 1000.0 / rep)  # ms total -> per-call us

    return {
        "size_label": size_label,
        "dtype": str(dtype),
        "correct": bool(correct),
        "max_abs_error": worst_err,
        "output_aliased_input": bool(aliased),
        "delegation": delegation,
        "n_blocks": n_blocks, "rep": rep, "warmup": warmup, "n_buffers": n_buffers,
        "latencies_us": latencies_us,
        "median_us": statistics.median(latencies_us) if latencies_us else 0.0,
        "mean_us": statistics.fmean(latencies_us) if latencies_us else 0.0,
        "stdev_us": statistics.pstdev(latencies_us) if len(latencies_us) > 1 else 0.0,
    }


def run_profile(kernel_fn: Callable, config: dict, seed: int = 42):
    device = "cuda"
    gen_fn = config["input_generator"]
    sizes = config["test_sizes"]

    prof_size = None
    for label, sz in sizes:
        if label == "medium":
            prof_size = sz
            break
    if prof_size is None:
        prof_size = sizes[0][1]

    dtype = config["test_dtypes"][0]
    inputs = gen_fn(prof_size, dtype, device, seed=seed)

    trace_dir = os.environ.get("CCO_TRACE_DIR", "./traces")
    os.makedirs(trace_dir, exist_ok=True)

    print("\n=== PROFILING ===")
    print(f"Profiling with size: {prof_size}, dtype: {dtype}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            kernel_fn(**inputs)
        torch.cuda.synchronize()
        for _ in range(10):
            kernel_fn(**inputs)
        torch.cuda.synchronize()

    trace_path = os.path.join(trace_dir, "kernel_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"profile_trace: {trace_path}")

    try:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# =========================================================================
# 5. MAIN
# =========================================================================

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="CCO benchmark harness")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel type to benchmark (default: read from kernel.py)")
    parser.add_argument("--sizes", type=str, default="all",
                        help="Which sizes to benchmark: small|medium|large|all (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip correctness stages 3-5, bench only large size")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch profiler trace")
    parser.add_argument("--json", action="store_true",
                        help="Emit one JSON metrics object to stdout; human-readable output goes to stderr")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for input generation. Self-score: 42 (default). Canonical "
                             "rerun: derive from the PR-HEAD SHA via cco/seed.py so inputs are unpredictable.")
    parser.add_argument("--score", action="store_true",
                        help="Emit the competition latency SAMPLE (n_blocks block-means) on the primary "
                             "size, with fused correctness + buffer rotation, for the significance test.")
    parser.add_argument("--blob", action="store_true",
                        help="Emit the bound score blob (sample + correctness + identity hashes + "
                             "blob_sha256) the gate pipeline verifies. Implies the scored sample.")
    args = parser.parse_args()
    json_stdout = sys.stdout
    if args.json:
        sys.stdout = sys.stderr

    # ------------------------------------------------------------------
    # Import the kernel module
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CCO Benchmark Harness")
    print("=" * 60)

    kernel_module = None
    kernel_fn = None
    kernel_type = args.kernel

    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())

        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn

        if kernel_type is None:
            kernel_type = getattr(kernel_module, "KERNEL_TYPE", None)
            if kernel_type is None:
                print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
                sys.exit(1)

        print(f"kernel_type: {kernel_type}")
        print("kernel_module: kernel.py loaded successfully")

    except SyntaxError as e:
        print("\nERROR: kernel.py has a syntax error:")
        print(f"  {e}")
        traceback.print_exc()
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        sys.exit(1)
    except Exception as e:
        print("\nERROR: Failed to import kernel.py:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        sys.exit(1)

    if kernel_type not in KERNEL_CONFIGS:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'")
        print(f"  Available: {', '.join(KERNEL_CONFIGS.keys())}")
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        sys.exit(1)

    config = KERNEL_CONFIGS[kernel_type]

    # ------------------------------------------------------------------
    # GPU Detection
    # ------------------------------------------------------------------
    gpu = detect_gpu()

    print("\n=== GPU INFO ===")
    print(f"gpu_name: {gpu.name}")
    print(f"gpu_sm_count: {gpu.sm_count}")
    print(f"gpu_memory_gb: {gpu.memory_gb}")
    print(f"gpu_peak_tflops_fp16: {gpu.peak_tflops_fp16}")
    print(f"gpu_peak_tflops_bf16: {gpu.peak_tflops_bf16}")
    print(f"gpu_peak_tflops_fp32: {gpu.peak_tflops_fp32}")
    print(f"gpu_peak_bandwidth_gb_s: {gpu.peak_bandwidth_gb_s}")
    print(f"gpu_l2_cache_mb: {gpu.l2_cache_mb}")
    print(f"gpu_compute_capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    print("\n=== CORRECTNESS ===")
    try:
        correctness_results = run_correctness(kernel_fn, config, quick=args.quick, seed=args.seed)
    except Exception as e:
        print(f"\nFATAL: Correctness testing crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        correctness_results = {"correctness": "FAIL", "smoke_test": "CRASH", "shape_sweep": "CRASH",
                               "numerical_stability": "CRASH", "determinism": "CRASH", "edge_cases": "CRASH"}

    print("\n--- Correctness Summary ---")
    print(f"smoke_test: {correctness_results.get('smoke_test', 'N/A')}")
    print(f"shape_sweep: {correctness_results.get('shape_sweep', 'N/A')}")
    print(f"numerical_stability: {correctness_results.get('numerical_stability', 'N/A')}")
    print(f"determinism: {correctness_results.get('determinism', 'N/A')}")
    print(f"edge_cases: {correctness_results.get('edge_cases', 'N/A')}")
    print(f"correctness: {correctness_results['correctness']}")

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    _perf_sizes = config["test_sizes"]
    _perf_primary_label = None
    _perf_primary_size = None
    for _pl, _ps in _perf_sizes:
        if _pl == "large":
            _perf_primary_label = _pl
            _perf_primary_size = _ps
            break
    if _perf_primary_size is None:
        _perf_primary_label, _perf_primary_size = _perf_sizes[-1]
    _perf_dtype = config["test_dtypes"][0]
    _size_params = ", ".join(f"{k}={v}" for k, v in _perf_primary_size.items())
    print(f"\n=== PERFORMANCE ({_perf_primary_label}: {_size_params}, dtype={_perf_dtype}) ===")

    perf_results = {"primary": None, "all": []}
    peak_vram_mb = 0.0
    try:
        sizes_filter = args.sizes
        if args.quick:
            sizes_filter = "large"
        torch.cuda.reset_peak_memory_stats()
        perf_results = run_performance(kernel_fn, config, gpu, sizes_filter=sizes_filter, seed=args.seed)
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception as e:
        print(f"\nFATAL: Performance benchmarking crashed: {type(e).__name__}: {e}")
        traceback.print_exc()

    primary = perf_results.get("primary")
    if primary is not None:
        print(f"\n--- Performance Summary (primary: {primary['label']}) ---")
        print(f"latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"throughput_tflops: {primary['throughput_tflops']:.3f}")
        print(f"bandwidth_gb_s: {primary['bandwidth_gb_s']:.1f}")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"arithmetic_intensity: {primary['arithmetic_intensity']:.2f}")
        print(f"ridge_point: {primary['ridge_point']:.2f}")
        print(f"bottleneck: {primary['bottleneck']}")
        print(f"flops: {primary['flops']}")
        print(f"bytes: {primary['bytes']}")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")

        print("\n=== COMPARISON VS PYTORCH ===")
        print(f"pytorch_latency_us: {primary['pytorch_latency_us']:.2f}")
        print(f"pytorch_latency_ms: {primary['pytorch_latency_us'] / 1000.0:.4f}")
        print(f"kernel_latency_us: {primary['kernel_latency_us']:.2f}")
        print(f"kernel_latency_ms: {primary['kernel_latency_us'] / 1000.0:.4f}")
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pytorch_tflops: {primary['ref_throughput_tflops']:.3f}")
        print(f"kernel_tflops: {primary['throughput_tflops']:.3f}")
    else:
        print("\nlatency_us: 0.00")
        print("latency_ms: 0.0000")
        print("throughput_tflops: 0.000")
        print("bandwidth_gb_s: 0.0")
        print("pct_peak_compute: 0.0%")
        print("pct_peak_bandwidth: 0.0%")
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")

        print("\n=== COMPARISON VS PYTORCH ===")
        print("pytorch_latency_us: 0.00")
        print("pytorch_latency_ms: 0.0000")
        print("kernel_latency_us: 0.00")
        print("kernel_latency_ms: 0.0000")
        print("speedup_vs_pytorch: 0.000x")

    # ------------------------------------------------------------------
    # Size sweep table
    # ------------------------------------------------------------------
    all_perf = perf_results.get("all", [])
    if len(all_perf) > 1:
        print("\n=== SIZE SWEEP ===")
        print(f"{'size':<12} {'kernel_us':>12} {'pytorch_us':>12} {'speedup':>10} {'tflops':>10} {'%peak':>8}")
        print("-" * 66)
        for entry in all_perf:
            print(f"{entry['label']:<12} {entry['kernel_latency_us']:>12.2f} "
                  f"{entry['pytorch_latency_us']:>12.2f} {entry['speedup_vs_pytorch']:>9.3f}x "
                  f"{entry['throughput_tflops']:>10.3f} {entry['pct_peak_compute']:>7.1f}%")

    # ------------------------------------------------------------------
    # Scored sample (competition; --score / --blob)
    # ------------------------------------------------------------------
    score_sample = None
    if args.score or args.blob:
        try:
            score_sample = run_scored_sample(kernel_fn, config, seed=args.seed)
        except Exception as e:
            print(f"score_error: {type(e).__name__}: {e}")
            traceback.print_exc()
        if args.score and score_sample is not None:
            sc = score_sample
            print("\n=== SCORE ===")
            print(f"score_size: {sc['size_label']}")
            print(f"score_dtype: {sc['dtype']}")
            print(f"score_correct: {sc['correct']}")
            print(f"score_delegation: {sc.get('delegation')}")
            print(f"score_output_aliased_input: {sc['output_aliased_input']}")
            print(f"score_max_abs_error: {sc['max_abs_error']:.6e}")
            print(f"score_median_us: {sc['median_us']:.4f}")
            print(f"score_mean_us: {sc['mean_us']:.4f}")
            print(f"score_stdev_us: {sc['stdev_us']:.4f}")
            print(f"score_n_blocks: {sc['n_blocks']}")
            print(f"score_latencies_us: {','.join(f'{x:.4f}' for x in sc['latencies_us'])}")

    # ------------------------------------------------------------------
    # Profiling (optional)
    # ------------------------------------------------------------------
    if args.profile:
        try:
            run_profile(kernel_fn, config, seed=args.seed)
        except Exception as e:
            print(f"\nWARNING: Profiling failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Final summary (greppable)
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start
    throughput = primary["throughput_tflops"] if primary else 0.0

    print("\n=== FINAL ===")
    print(f"kernel_type: {kernel_type}")
    print(f"input_seed: {args.seed}")
    print(f"correctness: {correctness_results['correctness']}")
    print(f"throughput_tflops: {throughput:.3f}")
    if primary:
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"bottleneck: {primary['bottleneck']}")
    else:
        print("speedup_vs_pytorch: 0.000x")
        print("pct_peak_compute: 0.0%")
        print("pct_peak_bandwidth: 0.0%")
    print(f"bench_time_seconds: {t_elapsed:.1f}")

    if t_elapsed > 90:
        print(f"WARNING: benchmark.py took {t_elapsed:.1f}s (budget: 90s)")

    if args.blob:
        from cco.blob import build_score_blob
        correctness_blob = {
            "overall": correctness_results.get("correctness"),
            "smoke_test": correctness_results.get("smoke_test"),
            "shape_sweep": correctness_results.get("shape_sweep"),
            "numerical_stability": correctness_results.get("numerical_stability"),
            "determinism": correctness_results.get("determinism"),
            "edge_cases": correctness_results.get("edge_cases"),
            "scored_correct": score_sample.get("correct") if score_sample else None,
            "scored_max_abs_error": score_sample.get("max_abs_error") if score_sample else None,
            "output_aliased_input": score_sample.get("output_aliased_input") if score_sample else None,
            "delegation": score_sample.get("delegation") if score_sample else None,
        }
        scored_blob = None
        if score_sample is not None:
            scored_blob = {k: score_sample[k] for k in (
                "size_label", "dtype", "n_blocks", "rep", "warmup",
                "median_us", "mean_us", "stdev_us", "latencies_us")}
        gpu_blob = {
            "name": gpu.name,
            "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
            "sm_count": gpu.sm_count,
            "memory_gb": gpu.memory_gb,
        }
        blob = build_score_blob(
            competition="CCO", version=1, kernel_type=kernel_type, seed=args.seed,
            correctness=correctness_blob, scored=scored_blob, gpu=gpu_blob,
            repo_root=repo_root, harness_path=os.path.abspath(__file__),
            kernel_path=getattr(kernel_module, "__file__", None),
        )
        print("\n=== SCORE BLOB ===")
        print(json.dumps(blob, sort_keys=True, indent=2))
        print("=== END SCORE BLOB ===")

    if args.json:
        metrics = _build_json_metrics(
            kernel_type,
            correctness_results,
            primary,
            perf_results,
            gpu,
            peak_vram_mb,
            t_elapsed,
        )
        _emit_json(metrics, json_stdout)


if __name__ == "__main__":
    main()
