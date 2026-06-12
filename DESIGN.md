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

**Isolation.** The authoritative scored run executes `kernel.py` in a **separate subprocess**
(`cco/isolate.py`). A kernel that shared the scorer's interpreter could otherwise read the secret
seed from `sys.argv`, monkeypatch `torch.allclose` / the timing events, or neuter the no-delegation
trap — no in-process guard can stop code that runs *inside* the interpreter. So the parent generates
the inputs from the secret seed (which **never enters the child**) and judges correctness with **its
own** oracle against the outputs the child returns — validating distinct buffers drawn both *before
and after* the timed window, so a correct-then-garbage call-counter has no safe window. The child is
launched with `-E` and a clean working directory (no `sitecustomize` / `PYTHON*` injection), and its
output is deserialized tensor-only (`weights_only=True`) so it cannot pickle-RCE the parent. The
in-child dispatch trap wraps the **entire scored window** — pre-validation, warmup, the timed loop,
and post-validation — so there is no untrapped phase in which a kernel could detect it is unobserved
(by catching `DelegationError`) and delegate to a fast vendor op only while being timed; a banned op
*anywhere* in that window is caught. The static guard's denylist is kept aligned with the runtime
trap's, and a submission may not import the `cco` package (so it cannot reach the trap internals).
Timing primitives (`torch.cuda.Event` / `synchronize` / `perf_counter`) are captured as child-locals
**before** the submission is loaded, so a kernel that monkeypatches them at import cannot forge its
timing. *Residual:* a genuinely-correct kernel under-reporting its own CUDA-event timing via
side-stream tricks is bounded by a captured-clock wall anchor on the sample's scale (events
implausibly faster than the wall are rejected); full timing-forge immunity needs parent-driven
two-point wall-clock timing, a planned follow-up.

**Timed-loop integrity.** The timed loop runs on a separate buffer set whose input is mutated by one
element before every call, so a kernel that memoizes by pointer must return a now-stale output and a
kernel that memoizes by content must recompute (honest timing). A sample of timed calls has its
`(input, output)` captured and oracle-checked — and the sampled positions are drawn from **server-side
entropy chosen at scoring time** (not the PR-HEAD seed, which the miner can recompute; not a
closed-form schedule), spread across **all** blocks so they overlap the median-feeding calls. Keeping
the schedule **unreadable** to the kernel — which shares the scorer's interpreter — is the load-bearing
requirement. The probe state lives in the timed loop's frame; reaching it from the kernel needs a walk
up the call stack (`f_back`), and *every* way to name a frame attribute is closed: the literal names
(`__traceback__` / `tb_frame` / `f_back` / `f_locals` / `gi_frame` / `__code__` / `__closure__` …) are
banned attributes, and the *string-keyed* routes that defeat a literal-name scan — `getattr`, the
`operator` module (`attrgetter`/`methodcaller`, so `operator` is not importable), and `str.format`
field access — are banned too, which makes frame-walking inexpressible rather than merely named. The job
file is also deleted before the kernel loads (`open` is banned), the kernel is invoked through a
module-level trampoline, and the CUDA allocator readout (`torch.cuda.memory_*`) is banned so the extra
buffer cloned at a probed call leaks nothing. A schedule-aware kernel therefore cannot do real work only
where it is observed: it must be correct on **every** timed call or risk the probe. An **absolute
roofline floor** (`max(bytes/peak_bw, flops/peak_flops)`) plus a coarse captured-clock wall anchor (event
median ≥ 0.5× the per-iter wall — a real side-stream offload drives that ratio toward zero, while honest
kernels sit near 1.0, so the threshold stays below the honest noise floor to never false-reject) reject
any physically-impossible or grossly side-stream-under-reported median;
creating CUDA streams/events/graphs (the side-stream channel) is itself statically banned. The static
guard is **re-run inside the scoring subprocess on the exact bytes about to execute**, so none of these
bans can be skipped by reaching the timed run unscanned. Together these close memoize-and-replay
(including the padded variant), the predictable-probe-schedule evasion, and the frame-introspection
schedule leak (literal *and* string-keyed).

**Native no-delegation backstop (the load-bearing guard).** The in-child dispatch trap is necessary
but not sufficient: a kernel sharing the interpreter can pop the TorchFunction/TorchDispatch mode
stack, delegate to a vendor GEMM, and push it back — all within one call, invisible to Python. So the
scoring child is launched with an **`LD_PRELOAD` shim** (`runtime/cco_preload.c`) that interposes the
vendor **compute** symbols (cuBLAS/cuBLASLt GEMM, cuDNN convolution/graph-execute, …) by name.
PyTorch links those libraries into the global symbol scope, so any `torch`→cuBLAS call — even one the
popped Python trap never saw — resolves to the shim, which records the symbol and `_exit(99)`s the
child; the parent reports it as delegation. The shim interposes *compute* entry points only (never
handle/descriptor setup), so a legit Triton kernel (which launches its own MMA via the CUDA driver)
and torch's own context init never trip it. The parent is **not** preloaded (it computes the cuBLAS
oracle), and it **refuses to score** unless a plain `torch.mm` child trips the shim — turning any
future linkage regression into a hard stop, not a silent universal bypass. *Residual:* a vendor
kernel statically compiled into `libtorch` that crosses no cuBLAS/cuDNN symbol (flash / mem-efficient
SDPA, row-wise fp8, int4-pack) is shim-blind; those remain guarded only by the static AST ban + the
(poppable) in-Python trap — hardening them is Tier-3 work.

## 5. Threat model — what's gameable, and what closes it

| Attack | Closed by | Residual? |
|---|---|---|
| Edit the harness / oracle / config / a champion | Gate 2 manifest re-hash (main-authoritative) | no |
| Inject a new file into a locked dir (auto-import RCE) | Gate 2 full-directory-listing pin | no |
| Delegate to `torch.matmul` / `F.*` / `@` / cuBLAS | static AST guard (Gate 3) **+** runtime dispatch trap (Gate 4) **+** `LD_PRELOAD` vendor-symbol trap | hand-rolled MMA vs "morally cuBLAS" is a policy line; Triton-only v1 shrinks it |
| Pop the in-Python trap mid-call, then delegate to a vendor GEMM | the `LD_PRELOAD` shim interposes the cuBLAS/cuBLASLt/cuDNN **compute** symbols in the scoring child (op-name-agnostic, unreachable from Python); harness refuses to score if it ever goes inert | fused kernels statically linked into `libtorch` (flash SDPA, row-wise fp8, int4-pack) cross no vendor symbol → guarded only by the static ban (Tier 3) |
| Delegate via an alternate GPU library (`cupy`/`jax`/`cutlass`/`numba`) or `torch.compile`/Inductor codegen | imports are an **allowlist** (only torch/triton/stdlib-utils) + the codegen namespaces (`torch.compile`/`_dynamo`/`_inductor`/`fx`/`jit`) are banned (Gate 3) | a GEMM that routes through cuBLAS is still caught by the shim; a CUTLASS kernel compiled into `libtorch` is the Tier-3 residual above |
| Dynamic dispatch to a banned op (aliasing, `getattr`/`eval`, `torch.__dict__['matmul']`) | builtins ban + introspection-dunder ban + import-alias resolution in the static guard | no (any GEMM that does execute still hits the runtime trap + the shim) |
| Walk the call stack to read the secret probe schedule (`e.__traceback__.tb_frame.f_back.f_locals`) | reaching a frame object REQUIRES naming an attribute (`tb_frame`/`gi_frame`/`__traceback__`); the guard bans every way to name one: literal `ast.Attribute`, `getattr` (builtin, incl. passed-as-value), the `operator` module (not importable), `str.format` field access, AND structural-pattern `match: case object(f_back=x)` (`MatchClass.kwd_attrs`); job-file deletion + trampoline + Gate-4 re-scan as backstops | **structural-completeness bet** — this assumes the five forms above are *all* the attribute-by-name routes; that bet has been wrong before (each round found the next form). The durable fix is not more enumeration but **thread-isolating the kernel** so the schedule lives in a frame its thread cannot reach (cross-thread frame access — `sys._current_frames`/`gc`/`threading` — is import-banned) |
| Infer the probe schedule from the extra buffer cloned at a probed call | `torch.cuda.memory_allocated`/`memory_stats`/`mem_get_info`/… are statically banned, so the kernel cannot read allocator deltas; it also cannot time itself (no `Event`, no `time` import) | low-severity even unbanned (allocator churn is noisy); banned outright regardless |
| Under-report timing via a side CUDA stream / graph (correct work, fast events) | static ban of the stream/event/graph API under **all** its qualified names — `torch.cuda.Stream`/`Event`/`CUDAGraph`, the canonical `torch.cuda.streams.*`, and the top-level `torch.Stream`/`torch.Event` (Triton uses the current timed stream — the PRIMARY closure) **+** a coarse captured-clock wall anchor (event median ≥ 0.5× per-iter wall; a real offload drives the ratio toward 0) | same enumeration risk as the row above (a stream alias under yet another path); the durable fix is **parent-driven two-point wall timing** (v2), which makes timing forge-resistant without enumerating stream names. A contrived *partial* offload on a Gate-3-bypassed run that parks the ratio above the anchor is the residual until then |
| Reach the scored GPU run without the static guard (Gate 3 → Gate 4 gap / TOCTOU) | the static guard is **re-scanned inside the scoring subprocess on the exact bytes about to `exec`**; any violation aborts as a delegation result before the kernel loads | no |
| Inline CUDA-C escape | banned in v1 (guard rejects `cpp_extension`) | n/a in v1 |
| Memorize / hardcode outputs for known inputs | PR-HEAD-seeded inputs the kernel **never sees** (process isolation); oracle re-derives truth | no |
| Cache first answer, return it always | parent-validates distinct buffers before + after timing | no |
| Memoize-and-replay (per-buffer cache → ~free timed loop) | per-call input mutation + kernel-unknowable timed-output probe + absolute roofline floor (§4) | no |
| Fast garbage only at the scored size | parent-validates the scored-size outputs against its oracle | no |
| Win via warm-L2 residency | rotating input buffers across reps | reduced; canonical box also locks clocks |
| Return a view of the input (no compute) | parent-side oracle validation (an unchanged input fails the oracle) | no |
| In-process score forgery (patch the comparison/timing, read the seed from `argv`) | the kernel runs **isolated** in a subprocess; the parent judges correctness with its own oracle and bounds timing by wall-clock (§4) | timing under-report (bounded) |
| Pickle-RCE the scorer from the subprocess | child output loaded `weights_only=True` (tensors only) | no |
| `os.system` / `import sys` / sitecustomize escape | static import ban (`os`/`sys`/`builtins`/`io`) + child `-E` + clean cwd | no for the verdict |
| OOM-dodge a locked size | OOM on a correctness size is a **FAIL**, not a skip | no |
| Pass intermittently / race conditions | within-tolerance determinism + multi-buffer correctness; gross races fail smoke/sweep | rare 1-in-10⁶ faults policed post-merge |
| Approximate/degraded output under loose tolerance | per-track locked tolerances (tightened; e.g. swiglu 0.5 → 0.01/0.2) | tolerance is a benchmark-validity knob; per-output tolerances are a noted refinement |
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
