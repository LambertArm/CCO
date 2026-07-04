"""Correctness tests for the tiling algorithm (GPU / PyTorch).

CCO computes on the GPU only, so these tests need a CUDA or Apple-MPS device;
they skip cleanly when none is present. They validate the *blocking math* —
ragged (non-divisible) tiles and fp16/fp32/fp64 accumulation.

Run:  python -m pytest tests/ -q      (or)   python tests/test_correctness.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()

# Under pytest, skip the whole module cleanly when no GPU is present — otherwise
# the GPU-gated imports below never run and the test bodies hit undefined names.
# (The __main__ runner does its own skip for `python tests/test_correctness.py`.)
try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not HAVE_GPU, reason="no CUDA/MPS GPU; CCO computes on GPU only")
except ImportError:
    pass

if HAVE_GPU:
    from matmul import matmul
    from matmul.backend import Backend
    from matmul.config import Config
    from matmul import gemm


def _run_tiled(n, T, dtype="fp32"):
    cfg = Config(dtype=dtype, tile=T, verbose=False)
    backend = Backend(verbose=False)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(cfg.np_dtype)
    B = rng.standard_normal((n, n)).astype(cfg.np_dtype)
    C = np.zeros((n, n), dtype=cfg.np_dtype)
    # Force the tiled path even though it fits in core.
    gemm._gemm_tiled_sync(A, B, C, backend, cfg, T)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    return C.astype(np.float64), ref


def test_tiled_divisible_fp32():
    C, ref = _run_tiled(64, 16, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_tiled_ragged_fp32():
    # n not a multiple of T -> exercises ragged edge tiles.
    C, ref = _run_tiled(100, 32, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_tiled_ragged_fp64():
    C, ref = _run_tiled(97, 40, "fp64")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-12


def test_tiled_fp16_accumulates_fp32():
    C, ref = _run_tiled(80, 24, "fp16")
    # fp16 inputs -> larger tolerance, but fp32 accumulation keeps it bounded.
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 5e-2


def test_fp16_incore_tiled_parity():
    """In-core and tiled must agree when accumulate_fp32=True (default)."""
    n, T = 256, 64
    cfg = Config(dtype="fp16", accumulate_fp32=True, verbose=False)
    backend = Backend(verbose=False)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(cfg.np_dtype)
    B = rng.standard_normal((n, n)).astype(cfg.np_dtype)

    C_incore = np.zeros((n, n), dtype=cfg.np_dtype)
    gemm._gemm_in_core(A, B, C_incore, backend, cfg)

    C_tiled = np.zeros((n, n), dtype=cfg.np_dtype)
    gemm._gemm_tiled_sync(A, B, C_tiled, backend, cfg, T)

    denom = np.linalg.norm(C_incore.astype(np.float64))
    rel = np.linalg.norm(
        (C_incore - C_tiled).astype(np.float64)) / max(denom, 1e-12)
    assert rel < 1e-6


def test_tile_larger_than_n():
    # T >= n must degenerate to a single block and still be correct.
    C, ref = _run_tiled(50, 128, "fp32")
    assert np.linalg.norm(C - ref) / np.linalg.norm(ref) < 1e-4


def test_public_matmul_matches_numpy():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    B = rng.standard_normal((128, 128)).astype(np.float32)
    C = matmul(A, B, config=Config(dtype="fp32", verbose=False))
    assert np.allclose(C, A @ B, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    if not HAVE_GPU:
        print("SKIP  all tests (no CUDA/MPS GPU; CCO computes on GPU only)")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
