
# Cuda-Compute-OSS
<img width="1104" height="517" alt="CCO-adver" src="https://github.com/user-attachments/assets/c34608c5-2370-4d86-a05c-6f01a93d9f5f" />




**An open arena for cheaper matrix multiplication.** `C = A × B` is the hot loop
of nearly all numerical computing. CCO is a place to submit *strategies* that
compute it for **less compute cost** — lower latency, lower VRAM, lower
time-complexity — **without giving up accuracy**, and to have that claim
measured the same way for everyone.

- **Normal (exact)** engine — the frontier you must beat: [`matmul/`](matmul/)
- **Smart (approximate)** strategies — where you contribute: [`strategy/`](strategy/)
- **The scorer** — one honest number per strategy: [`eval/`](eval/)
- **Landing page** — the project overview site: [`index/index.html`](index/index.html)

Reference setup: **`12000 × 12000`** matrices, **full-rank** (random) data, on an
**A100 (80 GB)** GPU via PyTorch.

```bash
python -m eval --n 12000 --pairs 3        # full-rank is the default
```

---

## Why CCO Exists

A faster matrix multiply is worth almost nothing if you can't **trust** it, and
almost every "10× faster" claim quietly pays for its speed somewhere you weren't
looking — a coarser dtype, a cherry-picked matrix, an unmeasured memory spike, a
benchmark that rewards being wrong quickly.

CCO exists to make the trade-off **visible and non-negotiable**:

- One **exact baseline** (`matmul/`) that every submission is measured against on
  the *same inputs*.
- One **scorer** (`eval/`) that reports accuracy, latency, peak VRAM and time
  complexity together — never one without the others.
- One **rule**: you may only claim an improvement when you reduce cost **and**
  hold accuracy. Trading accuracy for speed is not an improvement; it is a
  different, worse answer.

The goal is a growing library of strategies whose wins are real because they
were all forced through the same gate.

## How CCO Works

A **strategy** turns the exact `O(N³)` product into a cheaper approximate one.
The reference strategy is *subspace* multiplication: compress `A, B` into an
`M`-dimensional subspace (`M ≪ N`), multiply the small `M×M` matrices, and
reconstruct — `O(N²M)` instead of `O(N³)`. The subspace basis comes from a
pluggable **transform** ([`strategy/transforms.py`](strategy/transforms.py)),
which is the part you innovate on.

```
 A,B ──[ transform → basis Q ]──▶ compress (N,N)→(M,M) ──▶ multiply (M,M) ──▶ reconstruct (N,N) ≈ C
```

> **The bar is deliberately hard.** The reference regime is **full-rank** `12000`
> data — the general case, with no low-rank structure to exploit. A subspace of
> `M ≪ N` cannot capture a full-rank product, so the reference subspace strategy
> **does not beat exact here** (its accuracy collapses below any floor). That
> is the honest starting point: a strategy only scores by genuinely reducing cost
> *while holding accuracy* — on this data that means finding structure the
> reference `rsvd` basis misses (and adding it as a new transform). Compressible
> / low-rank inputs are the easy case where the subspace method already wins
> (see [BENCHMARKS.md](BENCHMARKS.md)).

The loop for a contributor is deliberately small:

1. **Add a transform** (or a whole new strategy) — a few lines implementing a
   basis `Q` (see [CONTRIBUTING.md](CONTRIBUTING.md)).
2. **Self-score locally** — `python -m eval …` runs your strategy and the exact
   baseline on identical random couples and prints one scorecard.
3. **Submit a PR** — paste the scorecard. The numbers, not the prose, decide.

Every strategy is scored by:

| axis | meaning | better is |
|---|---|---|
| **accuracy** | How close the approximated matrix is to the exact matrix. A score of 1.0 means the result is exactly correct. A score close to 0 means the approximation is poor. | higher |
| **time complexity** | analytic `O(·)` + an empirically-fitted `N^p` | lower |
| **latency** | wall-clock seconds of the multiply (GPU-synchronized) | lower |
| **VRAM usage** | peak *incremental* GPU memory during the multiply | lower |

combined into a single ranking score:

```
score = accuracy × (1 / Peak_VRAM) × (1 / Latency)
# 0 unless admitted as an improvement: accuracy ≥ floor AND latency, VRAM and
# FLOPs all below the exact baseline (the dominance rule)
```

See [BENCHMARKS.md](BENCHMARKS.md) for exactly how each number is produced.

## Correctness Gates

Cost metrics are only meaningful **after** a strategy is admitted as correct. A
submission is measured, then gated; a strategy that fails any gate scores **0**,
no matter how fast:

- **Accuracy floor** — the bounded accuracy score (`1 − ‖Ĉ − C‖_F / ‖C‖_F`,
  clamped to `[0,1]`) must be ≥ the floor (`--min-accuracy`, **default 0.8**) for
  the target regime. Below it, `score = 0`. A strategy that is fast and tiny but
  inaccurate cannot win.
- **Same-inputs rule** — exact and smart products are computed on the *identical*
  generated couples, in the same dtype, in one run. No separate baselines.
- **No-regression rule** — a change may not increase error on the regimes it
  already passed.

The exact engine itself is gated by [`tests/`](tests/) (ragged tiles, fp16
accumulation) on the GPU, so the baseline every strategy is judged against is
itself verified.

## Anti-Shortcut Rules

The fast way to a good score is usually a lie. These are rejected on sight:

- **No accuracy laundering.** You cannot buy latency/VRAM with accuracy. An
  improvement requires cost to drop while accuracy holds — see the dominance
  rule in [BENCHMARKS.md](BENCHMARKS.md).
- **No teaching to the test.** No hardcoding, caching, or looking up the
  evaluation matrices, seeds, or products. A strategy must work on unseen
  couples drawn from the same distribution.
- **No hidden precision downgrade.** Report the dtype you ran. Accuracy and
  latency must come from the *same* run at the *same* precision.
- **No unmeasured memory.** Peak VRAM is the peak of the PyTorch caching
  allocator during the whole multiply (`max_memory_allocated` on CUDA) — every
  transient tensor and workspace PyTorch allocates, not just the result. Memory
  a library grabs *outside* PyTorch's allocator is not captured; you cannot
  exclude a transient spike that goes through PyTorch.
- **No micro-win aggregation.** Sub-threshold gains are not summed across sizes
  or regimes to manufacture a headline; each claim stands on one regime.
- **Numbers over narrative.** If the scorecard and the description disagree, the
  scorecard wins.

## Quick Start

CCO computes on a **GPU** (CUDA or Apple MPS) via **PyTorch** — there is no CPU
or CuPy backend. You need a GPU machine (reference: A100, 80 GB).

```bash
# 1. install (NumPy + PyTorch — see requirements.txt)
pip install -r requirements.txt

# 2. see the exact baseline work (n defaults to 12000)
python -m matmul --n 12000 --verify

# 3. run a smart strategy
python -m strategy --n 12000 --transform rsvd --verify

# 4. self-score all strategies on the reference regime: 12000, full-rank
#    (this is what you paste in a PR)
python -m eval --n 12000 --pairs 3

# 5. run the tests (they skip if no GPU is present)
python tests/test_correctness.py
python eval/tests/test_eval.py
python strategy/tests/test_subspace.py
```

Start at [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository Layout

```
CCO/
├── matmul/         normal (exact) engine — the O(N³) frontier to beat  [matmul/README.md]
├── strategy/       smart (subspace) strategies; add transforms here    [strategy/README.md]
│   └── transforms.py   the pluggable "core tech" you innovate on
├── eval/           the scorer: accuracy · latency · VRAM · complexity → score  [eval/README.md]
├── tests/          correctness gates for the exact engine
├── examples/       runnable usage snippets
├── README.md          you are here — why/how/gates/rules
├── CONTRIBUTING.md    the one rule · self-score locally · submit
├── BENCHMARKS.md      how every number is produced · honesty notes
├── CHANGELOG.md
├── LICENSE            MIT
├── requirements.txt
└── .github/
    └── PULL_REQUEST_TEMPLATE.md   the scorecard your PR must fill in
```

Each of `matmul/`, `strategy/`, `eval/` is standalone and has its own README.

## License

CCO is released under the [MIT License](LICENSE). © 2026 CCO contributors.

