# CCO — Cuda-Compute-OSS

<p align="center">
  <img src="docs/assets/cco-readme-banner.png" alt="CCO Cuda-Compute-OSS banner" width="100%">
</p>

<p align="center">
  <a href="https://discord.gg/kEHZ3wJuHM"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Join the CCO Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/Subnet-74%20gittensor-orange" alt="Bittensor SN74">
</p>

**An objective GPU-kernel optimization competition on Bittensor subnet 74 (gittensor).**

Miners submit one optimized CUDA/Triton kernel. A locked benchmark harness verifies correctness
against a PyTorch oracle and times it on real hardware; if it beats the current champion — faster,
still correct, no more VRAM — with statistical significance, it becomes the new champion and earns
TAO emissions while it holds the crown. There is no subjective review: a submission
clears the statistical bar or it doesn't.

You optimize one file (`kernel.py`); everything else is locked and byte-verified.

---

## The tracks

Five kernels — the building blocks of a transformer layer — chosen to span the GPU optimization
regimes (memory-bandwidth-bound vs tensor-core/compute-bound). Each is its own king-of-the-hill
ladder with its own winner label.

| Track | What it is | Bottleneck |
|---|---|---|
| `rms_norm` | RMS normalization | memory-bound |
| `matmul` | general matrix multiply (GEMM) | compute-bound (tensor cores) |
| `qkv_part_rope` | partial rotary position embedding | memory-bound |
| `swiglu_input_quant` | SwiGLU activation + FP8 blockwise quant (multi-output) | memory-bound |
| `dsa_forward` | causal GQA attention (FlashAttention) | compute-bound (tensor cores) |

Each track ships a real Triton **champion** baseline (`champions/<track>/kernel.py`) that miners
must beat. New tracks are additive (drop in an oracle + config + champion + label) — the harness
auto-discovers them.

---

## How it works

```
 miner: optimize kernel.py  ─►  sign + open PR (JSON payload)
                                     │
   ┌──────── automated gate pipeline (stateless gates; default verdict = reject) ────────┐
   │  identity (GitHub↔hotkey↔SN74)  →  manifest (only kernel.py may differ)              │
   │  no-delegation AST scan  →  CANONICAL RERUN on trusted GPU (PR-HEAD-seeded inputs)   │
   └───────────────────────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
   correctness gate (5 stages PASS)  +  speedup vs CHAMPION (Mann-Whitney + margin)  +  VRAM guard
                                                    ▼
        win → merge, set `cco-winner-<track>`, strip prior winner  →  emissions (king-of-the-hill)
        else → close PR, credibility decrement
```

The harness ([`benchmark.py`](benchmark.py)) makes **no** decision — it emits a bound,
tamper-evident **score blob** (sample + correctness + identity hashes). CCO's gate pipeline
compares the challenger's blob to the champion's with a nonparametric test. Details in
[DESIGN.md](DESIGN.md).

---

## Quick start (miners)

```bash
cp champions/rms_norm/kernel.py kernel.py   # start from the current champion of a track
# ... edit kernel.py (Triton only; no delegation) ...
uv run benchmark.py                     # full correctness + roofline (self-score seed=42)
uv run benchmark.py --score             # the competition latency sample
uv run benchmark.py --blob              # the bound score blob the canonical rerun verifies
uv run --no-project python cco/guard_kernel.py kernel.py   # confirm no delegation
```

Then sign and open a PR with the [payload](payload-schema.json). Full rules:
[CONTRIBUTING.md](CONTRIBUTING.md).

> **Environment:** the competition is Linux + CUDA + Triton. Triton has no Windows wheels, so on
> Windows use WSL2. The canonical scoring rerun runs on a pinned GPU SKU.

---

## Project status

Early development, targeting gittensor SN74. The scoring stack is built and validated on real
hardware; the on-chain wiring (gate-pipeline automation, attested compute) is in progress.

| Area | Status |
|---|---|
| Anti-cheat (no-delegation static guard + runtime trap, manifest integrity) | **Ready** |
| Locked scorer (`benchmark.py`: seeded inputs, fused-correctness latency sample, bound blob) | **Ready** |
| Significance / win decision (`cco/significance.py`) | **Ready** |
| 5 champion kernels (`champions/`) | **Ready** (GPU-validated) |
| Config + payload schema (`cco.config.json`, `payload-schema.json`) | **Ready** |
| Runtime image, GPU-attested rerun, on-chain gate automation | **In progress** |
| Benchmark/leaderboard numbers | **Awaiting the first canonical runs** |

---

## Repository layout

```
kernel.py                 # *** THE MINER FILE *** (KERNEL_TYPE + kernel_fn); the one mutable artifact
benchmark.py              # locked scorer: 5-stage correctness + roofline + --score / --blob
cco/                      # enforcement: guard_kernel, gate, dispatch_trap, manifest_tool, seed, significance, blob
references/               # locked pure-PyTorch correctness oracles (5)
kernel_configs/           # locked benchmark spec + input generation (5)
champions/<track>/kernel.py   # locked per-track champion baseline (the bar to beat)
runtime/                  # locked rerun-image recipe (image digest recorded at publish)
docs/                     # GPU optimization study guides (informational; not locked)
manifest.json             # Gate-2 file-integrity set        cco.config.json   # policy + gittensor config
payload-schema.json       # PR submission schema             DESIGN.md         # design + threat model
BENCHMARKS.md             # scoring methodology + seed-champion baselines
```

---

## Community

Discord: **[discord.gg/kEHZ3wJuHM](https://discord.gg/kEHZ3wJuHM)**. GitHub issues are the canonical
place for design discussion, bugs, and roadmap.

## License

MIT — see [LICENSE](LICENSE). Winning kernels are permissively licensed so they can ship into
production.
