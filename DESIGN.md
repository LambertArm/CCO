# CCO — Design, Scoring & Threat Model

How the competition is built, how a submission is scored, and why it's hard to cheat. For the
miner-facing rules see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 1. The split: locked substrate vs. variable artifact

The single most important design decision: **the runtime is fixed and shared; the submission is the
only variable, and it is fed in.**

- **Locked (byte-verified at every PR HEAD against `manifest.json`):** the harness `benchmark.py`, the
  correctness oracles `references/`, the benchmark spec + input generation `kernel_configs/`, the
  per-track champions `champions/`, the enforcement code `cco/`, the config, and the runtime image.
- **Variable:** exactly one file, `kernel.py` (`KERNEL_TYPE` + `kernel_fn`), bound by `kernel_sha256`.
- **Per-PR secret:** the input seed = a function of the PR HEAD SHA (`cco/seed.py`), unknowable to
  the miner in advance.

The automated gate pipeline runs off-repo against this locked substrate; this repo ships only the
substrate + the per-repo config/state it reads.

## 2. The gate walk (default verdict = reject)

A PR merges only on affirmative evidence. The gate pipeline short-circuits on the first failure:

1. **Parse** — the fenced JSON payload against `payload-schema.json`; malformed → reject.
2. **Gate 1 — identity** — GitHub ↔ hotkey ↔ hotkey on the SN74 metagraph ↔ payload
   signature verifies under the hotkey.
3. **Gate 2 — manifest integrity** — re-hash every path in `main:manifest.json` at the PR HEAD; only
   `kernel.py` may differ, and **no unlisted file may be added** to a locked directory
   (`cco/manifest_tool.py` pins the full directory listing, closing the `kernel_configs` auto-import
   RCE). Gate 2 additionally rejects any PR whose **git diff touches anything other than
   `kernel.py`** — this complementary diff rule is what covers `manifest.json` itself, `.github/`,
   the docs, and stray top-level files outside any locked path. The repo's own CI status is
   advisory; no gate consults it.
4. **Gate 3 — no-delegation static scan** — `cco/guard_kernel.py` AST-rejects high-level/vendor ops,
   the `@` operator, dynamic-dispatch escapes, inline CUDA-C, and `get_*` exports; requires a
   `@triton.jit` kernel. Cheap; runs before any GPU spend.
5. **Rate-limit** — 1 canonical rerun / hotkey / 24h.
6. **Gate 4 — canonical rerun** — on trusted GPU hardware, egress closed, on PR-HEAD-seeded inputs:
   run the 5-stage correctness gate + the scored latency sample, with the runtime no-delegation trap
   (`cco/dispatch_trap.py`) active, and emit the bound score blob.

The PR is **frozen** between gate-walk and merge (snapshot of head SHA + body hash); any drift closes
it. This removes the "pass clean, then push a backdoor before merge" window.

## 3. Scoring

**Correctness is a hard gate, never an axis.** All 5 `benchmark.py` stages must PASS against the locked
oracle at the locked tolerances: smoke, shape sweep, numerical stability, **within-tolerance**
determinism (admits correct atomics/split-K kernels — not bitwise), edge cases. Speed never buys
back correctness.

**The scored axis is speedup vs the current champion — not vs the PyTorch oracle.** The oracle is
the *correctness* spec only; scoring against it would be meaningless (for `matmul` the oracle is
cuBLAS — unwinnable and a delegation magnet; for the memory-bound kernels it's slow eager PyTorch —
the first fused kernel wins 10× and the ladder then plateaus). So the bar is the standing champion
kernel, re-run **fresh and interleaved** with the challenger in the same sealed job (shared thermal
state).

**The measurement** (`run_scored_sample`): on the primary size + dtype, `n_blocks=30` block-mean
latencies — a *sample*, not a single median, because the win test is statistical. Three anti-cheat
properties are baked into the timing:
- **rotating input buffers** across reps (distinct seeds → distinct values & storage) — defeats
  warm-L2 residency and memoize-by-pointer;
- **fused correctness on 2 distinct buffers** — a kernel that caches buffer-0's answer fails on
  buffer-1 ("fast garbage at the scored size");
- an **output-vs-input alias guard** — a kernel returning a view of its input is rejected.

**The win decision** (`cco/significance.py`, run by the gate pipeline): a one-sided
**Mann-Whitney U** test on the two latency samples — nonparametric, so it's robust to the bimodal
GPU clock-boost that makes a Welch t-test misfire — **plus** an effect-size margin. A challenger
wins only if it is *significantly* faster **and** faster by ≥ `min_improvement_pct`. This rejects
both noise-only flips and statistically-detectable-but-tiny wins, preventing crown-thrash. **VRAM**
is a non-regression guard band.

**Emissions** (`cco.config.json` → `gittensor_repo_config`): king-of-the-hill. `fixed_base_score`
+ `default_label_multiplier: 0` means only the PR currently holding `cco-winner-<track>` earns, and
the magnitude of a win doesn't change pay — you're rewarded for *taking and holding* the frontier.

## 4. The bound score blob

`benchmark.py --blob` emits a sorted-keys JSON blob (`cco/blob.py`) that *proves what produced a score*:
the latency sample + correctness verdict, plus `input_seed`, `kernel_sha256` (the artifact),
`reference_hash` (oracle + config — so a score can't be computed against a weaker/edited oracle),
`harness_self_hash` (the scorer's own source), the GPU/SKU, and `blob_sha256` over all of it. Under
attestation (§6) `blob_sha256` is bound into the quote; in the v1 trusted-box posture it's the
integrity hash tying a score to its evidence. The harness makes no keep/revert decision.

## 5. Threat model — what's gameable, and what closes it

| Attack | Closed by | Residual? |
|---|---|---|
| Edit the harness / oracle / config / a champion | Gate 2 manifest re-hash (main-authoritative) | no |
| Inject a new file into a locked dir (auto-import RCE) | Gate 2 full-directory-listing pin | no |
| Delegate to `torch.matmul` / `F.*` / `@` / cuBLAS | static AST guard (Gate 3) **+** runtime dispatch trap (Gate 4) | hand-rolled MMA vs "morally cuBLAS" is a policy line; Triton-only v1 shrinks it |
| Inline CUDA-C escape | banned in v1 (guard rejects `cpp_extension`) | n/a in v1 |
| Memorize / hardcode outputs for known inputs | PR-HEAD-seeded inputs; oracle re-derives truth | no (seed unknowable) |
| Cache first answer, return it always | fused correctness on 2 distinct buffers | no |
| Fast garbage only at the scored size | fused correctness on the scored inputs | no |
| Win via warm-L2 residency | rotating input buffers across reps | reduced; canonical box also locks clocks |
| Return a view of the input (no compute) | output-vs-input alias guard | no |
| Pass intermittently / race conditions | within-tolerance determinism + multi-buffer correctness; gross races fail smoke/sweep | rare 1-in-10⁶ faults policed post-merge |
| Approximate/degraded output under loose tolerance | per-track locked tolerances (tightened; e.g. swiglu 0.5 → 0.01/0.2) | tolerance is a benchmark-validity knob; per-output tolerances are a noted refinement |
| Self-report a fake score | the canonical rerun runs it; the blob is bound (`blob_sha256`) | no |
| Win on a faster GPU | the SKU is pinned + part of the locked "model" (a swap is a vN reset) | requires attested SKU for full strength (§6) |

## 6. Attestation (v1 vs v2)

A kernel competition needs a **GPU** under confidential compute, so a CPU-only TEE does not apply.
- **v1 (current posture):** the canonical rerun runs on a **trusted, pinned GPU host**, egress
  closed, clocks locked, exclusive GPU. This closes the cheating surface (CCO runs the PR's code
  itself); it defers third-party *auditability*.
- **v2:** route the rerun through GPU-attested confidential compute (a GPU TEE) so the published
  image lands in MRTD and `blob_sha256` binds into the quote — then anyone can verify the rerun was
  honest. This is the make-or-break infrastructure dependency.

## 7. What's intentionally not here

There is no in-repo optimization agent or knowledge base. CCO ships only the locked substrate and
the single mutable `kernel.py`: the optimization intelligence is the external contributors, each
submitting one artifact to a frozen, objective harness.
