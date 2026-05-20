# Contributing to CCO

Thanks for your interest in contributing. This project lives or dies on the quality of its kernels, benchmarks, and optimization knowledge — every PR that improves any of those three is welcome.

## Ways to Contribute

1. **Add a new kernel** to optimize
2. **Submit benchmark results** on hardware not yet covered
3. **Improve the agent protocol** ([program.md](program.md)) or knowledge base ([CUDA_OPTIMIZATION.md](CUDA_OPTIMIZATION.md))
4. **Fix bugs** in the bench harness, NCU wrapper, or run loop
5. **Improve documentation** — examples, walkthroughs, reference material

---

## Development Setup

```bash
# Clone
git clone https://github.com/zeokin/Cuda-OSS.git
cd Cuda-OSS

# Install with dev extras
uv sync --extra dev

# Validate environment
uv run tools/prepare.py
```

You will need:
- An NVIDIA GPU on the [tested list](README.md#hardware)
- CUDA Toolkit matching your driver
- `ncu` (Nsight Compute CLI) on PATH for profiling contributions
- Python 3.10+

---

## Adding a New Kernel

A kernel contribution needs four pieces in three places. Use an existing kernel (e.g. `rms_norm`) as a reference.

### 1. The baseline kernel — `kernels/<name>.py`

Must export:

```python
KERNEL_TYPE: str          # e.g. "rms_norm"

def kernel_fn(**inputs) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """The kernel under optimization. Agent will edit this."""

def get_inputs() -> dict:
    """Return one sample input dict for smoke tests."""

def get_flops() -> int:
    """Total FLOPs for a single invocation at the smoke-test size."""

def get_bytes() -> int:
    """Total bytes accessed (read + write) for a single invocation."""
```

This file must remain a **working baseline**. It is the "before" picture against which optimized versions are compared.

### 2. The reference implementation — `references/<name>.py`

Pure PyTorch, no Triton, no custom CUDA. Used by the bench harness to verify correctness. Should be the simplest, most obviously-correct implementation possible — readability over performance.

### 3. The benchmark config — `kernel_configs/<name>.toml` + `kernel_configs/<name>.py`

The TOML declares test cases:

```toml
[[shapes]]
name = "tiny"
M = 128
N = 256
dtype = "bf16"
atol = 1e-2
rtol = 1e-2

[[shapes]]
name = "production"
M = 4096
N = 5120
dtype = "bf16"
atol = 1e-2
rtol = 1e-2
```

The companion Python module provides callables matching the TOML:

```python
def input_generator(shape: dict) -> dict: ...
def reference_fn(**inputs) -> torch.Tensor: ...
def flops_fn(shape: dict) -> int: ...
def bytes_fn(shape: dict) -> int: ...
```

### 4. Validate end-to-end

```bash
cp kernels/<name>.py kernel.py
uv run tools/bench.py          # full pipeline must pass
uv run tools/bench.py --quick  # quick pipeline must pass
uv run tools/ncu_profile.py    # NCU should produce a valid report
```

Open the PR only after all three commands succeed on at least one supported GPU.

---

## Submitting Benchmark Results

We track public benchmark results in [BENCHMARKS.md](BENCHMARKS.md). To submit:

### Prerequisites

- The kernel must already exist in `kernels/` (open a separate PR first if you are adding a new one)
- You must run the **full** optimization loop, not just a one-shot manual edit
- Numbers must come from `tools/bench.py`, not a custom harness

### Required information in your PR

```markdown
## Benchmark submission

- **Kernel:** `rms_norm`
- **GPU:** NVIDIA H100 80GB HBM3
- **Driver:** 550.54.15
- **CUDA:** 12.4
- **Triton:** 3.1.0
- **Agent + model:** Claude Opus 4.7
- **Final git SHA:** `<sha>`
- **Accepted iterations:** 14
- **Baseline (ms):** 0.612
- **Optimized (ms):** 0.341
- **Speedup:** 1.79×
- **% of peak BW:** 87.4%
- **Token cost (input + output):** 412k / 38k
```

Attach your `workspace/results.tsv` to the PR (or paste the last 5 rows inline).

PRs adding rows to `BENCHMARKS.md` without reproducible artifacts will be asked for them before merge.

---

## Code Style

- Python ≥ 3.10, type hints encouraged but not required for kernel code
- `ruff` for lint — run `uv run ruff check .` locally (CI runs `ruff check .` directly, without `uv sync`, to keep the lint job under a second). 120 char line length, see `pyproject.toml`.
- No new top-level dependencies without discussion in an issue first
- Tools should emit **greppable `key=value` output**, not prose — the agent has to parse it

## PR Conventions

- One logical change per PR
- Title format: `feat(kernel): add foo_kernel` / `fix(bench): handle dtype mismatch` / `docs: ...`
- Reference any related issue in the description
- For kernel additions, include a short paragraph on what the kernel computes and where it's used (which model, which layer, which paper)

## Reporting Bugs

Open an issue with:

- GPU model, driver, CUDA version (paste `nvidia-smi` and `nvcc --version`)
- The exact command that triggered the bug
- Full error output (not summarized)
- A minimal repro if you can produce one

For correctness bugs in a kernel, include the input shapes that triggered the discrepancy.

## Security

Do not file security issues in the public tracker. Email the maintainer instead.

## License

By contributing you agree that your contribution is licensed under the MIT License (see [LICENSE](LICENSE)).
