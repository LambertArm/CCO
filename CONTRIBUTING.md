# Contributing to CCO — Miner Guide

**CCO is an objective GPU-kernel optimization competition on Bittensor subnet 74 (gittensor).**
You (a "miner") submit one optimized kernel; if it beats the current champion — faster, still
correct, no more VRAM — with statistical significance, it becomes the new champion and earns
emissions while it holds the crown. There is no subjective review: a submission clears the bar or
it doesn't.

This guide is the contributor's reference. Read it once and you'll know exactly what an acceptable
submission looks like. For the architecture and threat model, see [DESIGN.md](DESIGN.md).

---

## 1. The one rule

**You may change exactly one file: `kernel.py`.** Everything else — the harness (`benchmark.py`),
the correctness oracles (`references/`), the benchmark spec (`kernel_configs/`), the per-track
champions (`champions/`), the config and runtime — is **locked** and byte-verified at your PR HEAD
against `manifest.json` (Gate 2). A PR that touches anything else is rejected.

## 2. The kernel contract

`kernel.py` must export exactly two names:

```python
KERNEL_TYPE = "rms_norm"   # one of the 5 tracks; selects the oracle/config/champion you compete on

def kernel_fn(**inputs):   # the kernel under test; called as kernel_fn(**inputs)
    ...                    # returns a Tensor, or a tuple of Tensors for multi-output tracks
```

It must **not** export `get_inputs` / `get_flops` / `get_bytes` — inputs, FLOPs and bytes are owned
by the locked `kernel_configs/`. The exact `kernel_fn` signature and the inputs you receive are
defined per track:

| Track | `kernel_fn(...)` | returns |
|---|---|---|
| `rms_norm` | `(x, weight, eps=1e-6)` | `Tensor` |
| `matmul` | `(a, b)` | `Tensor` |
| `qkv_part_rope` | `(qkv, cos, sin, q_heads, kv_heads, nope_dim)` | `Tensor` |
| `swiglu_input_quant` | `(x)` | `(out, x_fp8, x_scale)` |
| `dsa_forward` | `(q, k, v, block_indices, indices_blk_siz, scale, cu_seqlens_q, cu_seqlens_k, …)` | `(out, lse)` |

The **current champion** at `champions/<track>/kernel.py` is the canonical, correct, working
example for each track — start from it. The matching `kernel_configs/<track>.py` shows the exact
input dict you'll receive.

## 3. No delegation — write a real kernel

The competition measures *the kernel you write*, not your ability to call a library. **v1 is
Triton-only.** `kernel_fn` (and anything it calls) may **not**:

- call `torch.matmul / mm / bmm / addmm / einsum`, `torch.nn.functional.*` (rms_norm, softmax,
  silu, scaled_dot_product_attention, …), `torch.ops.aten.*`, `torch._scaled_mm` / `_int_mm` /
  `_weight_int*pack_mm`, `torch.linalg.*`, or the `@` matmul operator — i.e. any vendor BLAS/DNN call;
- JIT/codegen the kernel: `torch.compile`, `torch._dynamo` / `torch._inductor` / `torch.fx` / `torch.jit`;
- reach any of the above through aliases, `getattr` / `eval` / `exec` / `importlib`, introspection
  dunders (`__dict__` / `__getattribute__` / `__class__` …), `open` (no file I/O), or tensor methods
  (`a.mm(b)`);
- use inline CUDA-C (`torch.utils.cpp_extension`), or pop/neuter the runtime trap
  (`torch.overrides`, `torch.utils._python_dispatch`);
- define `get_inputs`/`get_flops`/`get_bytes`.

**Imports are an allowlist:** only `torch`, `triton`, and a few pure-Python utilities
(`math` / `typing` / `dataclasses` / `functools` / `collections` / …). No `os` / `sys` / `ctypes` /
`subprocess` / `numpy` (it re-exposes ctypes), and **no alternate GPU-compute library**
(`cupy` / `jax` / `cutlass` / `numba` / …).

It **must** contain at least one `@triton.jit` kernel and do the actual compute there. Allowed in the
Python wrapper: allocation (`torch.empty`/`empty_like`), reshape/view/transpose/contiguous, dtype
casts, shape introspection, and launching your Triton kernel.

This is enforced **mechanically**, not by review, in three layers: a static AST guard
([`cco/guard_kernel.py`](cco/guard_kernel.py)) rejects delegation before any GPU spend; a runtime trap
([`cco/dispatch_trap.py`](cco/dispatch_trap.py)) catches it during execution; and a native
`LD_PRELOAD` vendor-symbol trap catches any cuBLAS/cuDNN call that slips past the first two — even one
reached by popping the Python trap. Don't try to wrap cuBLAS — you'll be caught.

## 4. Self-score locally

```bash
uv run benchmark.py                   # full 5-stage correctness + roofline (on the published self-score seed=42)
uv run benchmark.py --score           # the competition latency SAMPLE on the primary size
uv run benchmark.py --blob            # the full bound score blob the canonical rerun verifies
uv run --no-project python cco/guard_kernel.py kernel.py    # check you didn't delegate
```

Self-scoring uses the **published seed (42)** so you can iterate. The canonical rerun uses a secret
seed derived from your PR HEAD SHA, so you cannot precompute or memorize outputs — your kernel must
be genuinely general.

Self-scoring needs **Linux with a CUDA GPU** (on Windows, use WSL2): `pyproject.toml` installs
torch/triton only on Linux, and Triton has no Windows wheels. (No GPU? You can at least run the AST
guard, which is pure Python.)

## 5. Submit

1. Register a hotkey on SN74 and bind your GitHub identity to it.
2. Put your kernel in `kernel.py`, commit it (only `kernel.py` changed).
3. Open a PR using the template: the fenced **JSON payload** (`payload-schema.json`) + the
   acknowledgement checkboxes. Sign `<commit_sha>:<kernel_sha256>:<kernel_type>` with your hotkey.
4. The PR is **frozen** once the gates pass — any edit (even a typo fix) closes it; open a fresh PR.

## 6. How you're scored

CCO's automated gate pipeline walks cheap gates (identity → manifest → no-delegation static scan →
threshold), then runs a **canonical rerun** on trusted GPU hardware:

- **Correctness is a hard gate** — all 5 stages must PASS (smoke, shape sweep, numerical stability,
  within-tolerance determinism, edge cases) against the locked oracle on the secret-seeded inputs.
- **The scored axis is speedup vs the current champion** (not vs PyTorch). Champion and challenger
  are re-run fresh and interleaved; you win only if a **Mann-Whitney U** test says you're faster
  **and** you beat the champion by at least the configured margin. A sub-noise or below-margin win
  does not take the crown.
- **VRAM is a non-regression guard** — you can't win by blowing up scratch memory.

Emissions are **king-of-the-hill**: only the PR currently holding `cco-winner-<track>` earns, and a
fixed base score means you're paid for *holding* the frontier, not for the size of one win. When a
new winner lands, your label is stripped.

**Rate limit:** 1 canonical rerun / hotkey / 24h (a new PR resets the clock — PR spam is
negative-EV). Hotkeys that repeatedly fail the rerun lose credibility and hit a banlist.

## 7. Reporting bugs / security

Open an issue with GPU/driver/CUDA, the exact command, and the full error (don't summarize the
stack trace). For a **correctness** bug, include the shapes/dtypes and the `pct_within_tol` figure.
Do **not** file security issues publicly — use GitHub's private vulnerability reporting first; a leaked token in git history
must be revoked, not just rotated.

## 8. License

MIT (see [LICENSE](LICENSE)). By contributing you agree your contribution is MIT-licensed — CCO
depends on staying permissive so winning kernels can ship into production.
