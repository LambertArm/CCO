"""CPU-only tests for matmul tile VRAM budgeting (no GPU required).

Run:  python tests/test_auto_tile_budget.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config
from matmul.gemm import _tile_operand_bytes, _tile_workspace_bytes_per_elem, auto_tile


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


def test_tile_operand_bytes_fp16_upcast():
    fp16_acc = Config(dtype="fp16", accumulate_fp32=True)
    fp16_raw = Config(dtype="fp16", accumulate_fp32=False)
    fp32 = Config(dtype="fp32")

    assert _tile_operand_bytes(fp16_acc) == 4
    assert _tile_operand_bytes(fp16_raw) == 2
    assert _tile_operand_bytes(fp32) == 4


def test_tile_workspace_counts_fp16_upcast():
    fp16_acc = Config(dtype="fp16", accumulate_fp32=True)
    # acc fp32 (4) + two fp32 operand tiles (4 + 4)
    assert _tile_workspace_bytes_per_elem(fp16_acc) == 12

    fp16_raw = Config(dtype="fp16", accumulate_fp32=False)
    assert _tile_workspace_bytes_per_elem(fp16_raw) == 6


def _legacy_auto_tile(n: int, cfg: Config, free_bytes: int) -> int:
    """Pre-fix estimator that sized fp16 operand tiles at item_bytes."""
    budget = int(free_bytes * cfg.vram_fraction)
    per_elem = cfg.acc_dtype.itemsize + 2 * cfg.item_bytes
    t = int(math.sqrt(max(1, budget) / per_elem))
    t = min(t, n)
    t = max(128, (t // 128) * 128)
    return min(t, n)


def test_auto_tile_fp16_accumulate_smaller_than_legacy_estimator():
    """fp16+accumulate_fp32 must not inherit fp16 operand sizing in the budget."""
    free = 64 * 1024**2
    backend = _FakeBackend(free)
    n = 8192
    cfg = Config(dtype="fp16", accumulate_fp32=True, vram_fraction=0.6)

    t_fixed = auto_tile(n, cfg, backend)
    t_legacy = _legacy_auto_tile(n, cfg, free)
    assert t_fixed <= t_legacy
    assert t_fixed < t_legacy


if __name__ == "__main__":
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
