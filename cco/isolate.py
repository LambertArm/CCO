"""
cco/isolate.py — run a submitted kernel in an ISOLATED subprocess so it cannot forge its score.

THE THREAT. The submitted kernel runs with full Python privileges. If it shares an interpreter
with the scorer, it can read the secret input seed from `sys.argv`, monkeypatch `torch.allclose` /
the comparison / the timing events / even the no-delegation trap, or execute arbitrary code at
import (a repo-root `sitecustomize.py`, `os.system`, ...). An in-process guard cannot stop this —
the attacker is already inside the interpreter.

THE FIX. Move every TRUSTED operation into the PARENT, and run only `kernel_fn` in a clean child:

    parent: generate ALL stage inputs from the secret seed (NEVER sent to the child)
        │  serialize the input TENSORS (not the seed) to a temp file
        ▼
    child : python -E, clean cwd, kernel loaded by file path (no sys.path injection, no
            sitecustomize), runs kernel_fn under the dispatch trap on each input, times the scored
            size with CUDA events, and writes back the OUTPUT TENSORS it produced
        │
        ▼
    parent: validates EVERY returned output against ITS OWN oracle (the child's comparison is never
            trusted) across the full correctness suite — smoke, shape sweep (all sizes × dtypes),
            numerical stability, determinism, edge cases — plus the scored-size buffers checked
            before AND after the timed window. Timing is bounded by the parent's wall-clock.

What this CLOSES: in-process correctness forgery, seed-read, scorer-state monkeypatching, the
sitecustomize / PYTHON* / os.system paths affecting the verdict, OOM/crash on a locked size (= a
stage FAIL), garbage-on-replay, and memoize-and-replay (incl. the padded variant). The timed loop
mutates its input before every call so a content cache must recompute (honest timing); a parent
oracle-checks a kernel-unknowable sample of timed outputs so a pointer cache returning a stale answer
is caught; and an absolute roofline floor rejects any physically-impossible median. The child trap is
defense-in-depth; the load-bearing checks are the parent's oracle over the whole suite + the floor.

Cost of the mutation: one extra Triton-store launch per timed call adds a small, FIXED latency
(launch overhead, exposed only for very fast kernels). It is identical for the champion and the
challenger, so the relative win test is unbiased; it slightly raises the minimum *detectable*
improvement, which also damps crown-thrash on negligible wins.

Residual (documented): a kernel that is genuinely correct but under-reports its CUDA-event timing
from inside the child. The parent wall-clock tripwire + roofline floor reject the impossible cases
(claimed GPU time > wall, or below the hardware bound); full immunity needs parent-driven two-point
wall-clock timing, a follow-up.

Usage (needs a CUDA GPU + torch/triton; run in the Linux/WSL env):
    ~/cco-gpu/bin/python cco/isolate.py --self-test
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _to_cpu(out):
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cpu(o) for o in out)
    return out.detach().to("cpu")


def _to_cuda(out):
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cuda(o) for o in out)
    return out.to("cuda")


def _has_nan_inf(out) -> bool:
    import torch
    items = out if isinstance(out, (tuple, list)) else [out]
    for t in items:
        if torch.isnan(t.float()).any().item() or torch.isinf(t.float()).any().item():
            return True
    return False


# =====================================================================================
# CHILD — runs in the isolated subprocess. Untrusted-kernel territory; runs kernel_fn on each
# parent-provided input and returns the raw outputs. Makes NO judgement.
# =====================================================================================

def _child_main(job_path: str, out_path: str) -> int:
    import importlib.util

    import torch

    job = torch.load(job_path, weights_only=True)

    import time

    # Capture the timing primitives + trap as LOCALS *before* the submission is loaded. The kernel
    # shares this child interpreter, so it could monkeypatch torch.cuda.Event / torch.cuda.synchronize
    # / time.perf_counter at import to forge its own timing — but it cannot reach these function-local
    # captures or repatch them, and it cannot reach the timing loop (harness code), so the loop below
    # stays honest no matter what the kernel patches globally.
    _Event = torch.cuda.Event
    _sync = torch.cuda.synchronize
    _perf = time.perf_counter
    from cco.dispatch_trap import DelegationError, delegation_trap, run_guarded

    # The job (incl. the secret probe schedule) is now fully in memory — DELETE the file before the
    # submission loads so a kernel cannot read which timed calls are oracle-checked (`open` is also
    # statically banned; this is the runtime backstop). The schedule then lives only in this function's
    # locals, unreachable from the kernel's module namespace.
    try:
        os.remove(job_path)
    except OSError:
        pass

    spec = importlib.util.spec_from_file_location("cco_submission_kernel", job["kernel_path"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the only place the submission executes; this is the isolated child
    kernel_fn = mod.kernel_fn

    def cuda_in(inp):
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}

    def clone_gpu(o):  # async GPU clone (no host sync) — cheap enough to capture inside the loop
        if isinstance(o, (tuple, list)):
            return type(o)(clone_gpu(x) for x in o)
        return o.detach().clone()

    def clone_in(d):
        return {k: (v.detach().clone() if hasattr(v, "detach") else v) for k, v in d.items()}

    delegation = None
    child_error = None
    task_outputs = []
    det_outputs = []
    scored_val = []
    event_block_us: list[float] = []
    timed_wall_s = 0.0
    probe_in_gpu: list = []   # captured (input, output) of a parent-chosen, kernel-unknowable sample
    probe_out_gpu: list = []  # of TIMED calls; the parent oracle-checks them (defeats a stale cache)
    probe_inputs: list = []   # ^ the same, moved to CPU for the parent after the timed window
    probe_outputs: list = []

    try:
        # --- correctness tasks (smoke / sweep / stability / edge): run once each ---
        for t in job["tasks"]:
            if delegation:
                task_outputs.append({"output": None, "error": "skipped"})
                continue
            try:
                out = run_guarded(kernel_fn, cuda_in(t["inputs"]))
                task_outputs.append({"output": _to_cpu(out), "error": None})
            except DelegationError as e:
                delegation = str(e)
                task_outputs.append({"output": None, "error": "delegation"})
            except Exception as e:  # OOM / crash on a locked size is a per-task failure
                task_outputs.append({"output": None, "error": f"{type(e).__name__}: {e}"})
            torch.cuda.empty_cache()

        # --- determinism: run the same input N times ---
        if delegation is None and job.get("determinism"):
            d = job["determinism"]
            di = cuda_in(d["inputs"])
            for _ in range(int(d["runs"])):
                det_outputs.append(_to_cpu(run_guarded(kernel_fn, di)))
            torch.cuda.empty_cache()

        # --- scored size: EVERY call (pre-val / warmup / timed / post-val) runs under the trap, so
        #     there is no untrapped phase in which a kernel could delegate; a banned op ANYWHERE here
        #     raises DelegationError and is caught below.
        #     The TIMED loop runs on a SEPARATE buffer set whose content is MUTATED in place before
        #     every call (one element write, unique per call): a content-addressed cache misses and
        #     must recompute (honest timing), while a pointer-addressed cache returns a now-STALE
        #     output. To catch the latter, a parent-chosen, kernel-UNKNOWABLE sample of timed calls
        #     has its (mutated input, output) captured for the parent to oracle-check. Pre/post-val
        #     use the CLEAN buffers (never mutated), so the correct-then-garbage check is unaffected. ---
        if delegation is None:
            sc = job["scored"]
            bufs = [cuda_in(b) for b in sc["buffers"]]              # clean: pre/post validation only
            nb = len(bufs)
            n_pre = int(sc["n_pre"])
            rep = int(sc["rep"])
            n_blk = int(sc["n_blocks"])
            tbufs = [cuda_in(b) for b in sc["timed_buffers"]]       # separate storage: mutated while timing
            ntb = len(tbufs)
            mut_key = sc["mut_key"]
            probe_set = {tuple(p) for p in (sc.get("probe_positions") or [])}   # server-random (block, rep)

            # Per-call input mutation via a tiny TRITON store, NOT a torch op: a torch setitem would be
            # intercepted by the delegation trap (~tens of us of Python per call) and make the CPU, not
            # the kernel, the bottleneck for fast kernels. A Triton launch bypasses the torch dispatcher
            # — the trap never sees it — and the destination views are built once, outside the trap. It
            # writes a MONOTONICALLY growing value to element 0 before every timed call: the value at a
            # captured (late) call is therefore far from the value at the buffer's first touch, so a
            # pointer-cache that replays its first-touch output is stale in row 0 by a wide margin —
            # detectable at ANY track tolerance — while a content-cache must recompute (honest timing).
            # Element 0 (vs a rotating index) keeps the store a single un-specialized Triton kernel.
            import triton
            import triton.language as tl

            @triton.jit
            def _mut_k(PTR, VAL):
                tl.store(PTR, VAL)

            _flats = [tb[mut_key].view(-1) for tb in tbufs]
            _do_mut = (bool(_flats) and hasattr(tbufs[0].get(mut_key), "is_floating_point")
                       and tbufs[0][mut_key].is_floating_point())

            def mutate(buf_idx, gi):                                # 1 Triton launch, no trapped torch op
                if _do_mut:                                        # 37 + gi: monotonic, bounded < fp16 max
                    _mut_k[(1,)](_flats[buf_idx], float(37 + gi % 60000))

            g = 0                                                   # global timed-call index
            with delegation_trap() as _trap_verify:
                for i in range(min(n_pre, nb)):
                    scored_val.append(_to_cpu(kernel_fn(**bufs[i])))

                for _ in range(int(sc["warmup"])):
                    mutate(0, g)
                    g += 1
                    kernel_fn(**tbufs[0])
                _sync()
                s = _Event(enable_timing=True)
                e = _Event(enable_timing=True)
                _t0 = _perf()
                for blk in range(n_blk):
                    _trap_verify()         # between blocks: catch a kernel that popped the trap mid-loop
                    s.record()
                    for r in range(rep):
                        tb = tbufs[g % ntb]
                        mutate(g % ntb, g)
                        grab = (blk, r) in probe_set
                        in_clone = clone_in(tb) if grab else None   # the exact input handed to the kernel
                        out = kernel_fn(**tb)
                        if grab:
                            probe_in_gpu.append(in_clone)
                            probe_out_gpu.append(clone_gpu(out))
                        g += 1
                    e.record()
                    _sync()
                    # EVERY block feeds the median (probe positions are scattered across all blocks and
                    # unknowable to the kernel), so a kernel cannot garbage a median-feeding call without
                    # risking the oracle probe. The few async clones at probe positions are negligible.
                    event_block_us.append(s.elapsed_time(e) * 1000.0 / rep)
                # Captured-clock wall of the whole timed window: forge-resistant (the kernel cannot
                # patch _perf or _sync here), so the parent anchors the cuda-event sample's scale to
                # it — a kernel under-reporting events (e.g. side-stream evasion) is caught.
                timed_wall_s = _perf() - _t0

                for i in range(n_pre, nb):
                    scored_val.append(_to_cpu(kernel_fn(**bufs[i])))
                _sync()

            probe_inputs = [{k: (_to_cpu(v) if hasattr(v, "detach") else v) for k, v in p.items()}
                            for p in probe_in_gpu]
            probe_outputs = [_to_cpu(p) for p in probe_out_gpu]
    except DelegationError as ex:
        delegation = str(ex)
    except Exception as ex:
        # A crash (wrong signature, OOM, runtime error) in determinism/scored is a graceful FAIL,
        # not a child crash: leave the (incomplete) outputs and let the parent mark the missing
        # stages FAIL rather than losing the whole verdict.
        child_error = f"{type(ex).__name__}: {ex}"

    if delegation is not None:
        det_outputs, scored_val, event_block_us = [], [], []
        probe_inputs, probe_outputs = [], []

    torch.save({"task_outputs": task_outputs, "det_outputs": det_outputs,
                "scored_val": scored_val, "event_block_us": event_block_us,
                "timed_wall_s": timed_wall_s, "delegation": delegation,
                "probe_inputs": probe_inputs, "probe_outputs": probe_outputs,
                "child_error": child_error}, out_path)
    return 0


# =====================================================================================
# PARENT — trusted. Generates inputs from the secret seed, spawns the child, validates EVERY
# returned output against the oracle across the full correctness suite, bounds timing by wall-clock.
# =====================================================================================

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preload_so() -> "str | None":
    """Resolve the Tier-2 LD_PRELOAD vendor-symbol trap (runtime/cco_preload.so), if built.

    Honors $CCO_PRELOAD_SO; otherwise looks next to the repo's runtime/. Returns None when the .so is
    absent (e.g. on a dev box where it was not compiled, or Windows) — the in-Python trap still runs;
    the LD_PRELOAD backstop is an additive defense that the trusted GPU image always builds."""
    env = os.environ.get("CCO_PRELOAD_SO")
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    cand = os.path.join(_repo_root(), "runtime", "cco_preload.so")
    return cand if os.path.isfile(cand) else None


_PRELOAD_SELFTEST = None  # None=unchecked, True=verified, str=failure reason (cached per process)


def _assert_preload_interposes(preload: str) -> None:
    """FAIL-CLOSED gate: confirm the LD_PRELOAD trap actually interposes a real `torch.mm` on THIS
    host. If a plain matmul child does NOT _exit(99), interposition is broken (static-linked cuBLAS,
    a renamed symbol, an RTLD quirk, a stale .so) and EVERY delegation would pass silently — so we
    refuse to score rather than hand out a free pass. Runs once per process (cached)."""
    global _PRELOAD_SELFTEST
    if _PRELOAD_SELFTEST is True:
        return
    if isinstance(_PRELOAD_SELFTEST, str):
        raise RuntimeError(_PRELOAD_SELFTEST)
    import subprocess as _sp
    probe = ("import torch;"
             "a=torch.randn(64,64,device='cuda',dtype=torch.float16);"
             "b=(a@a); torch.cuda.synchronize()")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    existing = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    env["LD_BIND_NOW"] = "1"
    try:
        p = _sp.run([sys.executable, "-E", "-c", probe], env=env,
                    capture_output=True, text=True, timeout=180)
    except Exception as e:  # noqa: BLE001 - any failure to run the gate is fail-closed
        _PRELOAD_SELFTEST = f"LD_PRELOAD self-test could not run ({type(e).__name__}: {e}); refusing to score"
        raise RuntimeError(_PRELOAD_SELFTEST) from e
    if p.returncode == 99:
        _PRELOAD_SELFTEST = True
        return
    _PRELOAD_SELFTEST = (
        f"LD_PRELOAD vendor trap is INERT: a plain torch.mm child exited {p.returncode}, not 99 — "
        f"symbol interposition is broken on this host (static-linked cuBLAS / renamed symbol / stale "
        f".so). Refusing to score: a delegating kernel would pass undetected. "
        f"stderr tail: {(p.stderr or '')[-300:]}")
    raise RuntimeError(_PRELOAD_SELFTEST)


def run_isolated(kernel_path: str, config: dict, seed: int, compare_fn, *,
                 n_blocks: int = 30, warmup: int = 25, rep: int = 100,
                 n_val: int = 6, n_timed: int = 4, quick: bool = False, timeout_s: float = 1200.0,
                 peak_bw_gb_s: float = 0.0, peak_tflops: float = 0.0,
                 floor_fraction: float = 0.8) -> dict:
    """Score `kernel_path` isolated; judge correctness HERE against the oracle over the full suite.

    `compare_fn(output, expected, atol, rtol, multi_output) -> {"match": bool, "max_abs_error": ..}`
    is supplied by the caller (benchmark._do_compare) so this module stays torch-light at import.
    Returns the run_scored_sample dict shape + a `stages` dict (smoke_test/shape_sweep/.../correctness).
    """
    import random
    import statistics
    import time

    import torch

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    multi = config.get("multi_output", False)
    dtypes = config["test_dtypes"]
    sizes = config["test_sizes"]
    tols = config["tolerances"]
    edge_sizes = config.get("edge_sizes", [])
    dev = "cuda"

    def tol_for(dt):
        return tols.get(dt, {"atol": 1e-2, "rtol": 1e-2})

    # ---- build the task list in the PARENT: inputs (-> CPU for the child) + oracle (kept for check) ----
    tasks, specs = [], []

    def add_task(inputs_gpu, stage, dt, *, relax=1.0, both_nan_ok=False):
        expected = ref_fn(inputs_gpu)
        tasks.append({"inputs": {k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
                                 for k, v in inputs_gpu.items()}})
        t = tol_for(dt)
        specs.append({"stage": stage, "expected": _to_cpu(expected),
                      "atol": t["atol"] * relax, "rtol": t["rtol"] * relax, "both_nan_ok": both_nan_ok})

    # scored size = "large" or last
    size_label, scored_size = None, None
    for label, sz in sizes:
        if label == "large":
            size_label, scored_size = label, sz
            break
    if scored_size is None:
        size_label, scored_size = sizes[-1]

    # smoke (sizes[0] x dtypes[0])
    add_task(gen_fn(sizes[0][1], dtypes[0], dev, seed=seed), "smoke_test", dtypes[0])
    # shape sweep (all sizes x dtypes)
    for _lbl, sz in sizes:
        for dt in dtypes:
            add_task(gen_fn(sz, dt, dev, seed=seed), "shape_sweep", dt)

    # stability size (small or 2nd) + the adversarial transforms (parent applies them, trusted)
    stab_size = next((sz for lbl, sz in sizes if lbl == "small"), sizes[min(1, len(sizes) - 1)][1])
    if not quick:
        def _xf_near_max(t):
            return t * (60000.0 if t.dtype == torch.float16 else 1e30)
        transforms = [("near_max", _xf_near_max), ("near_zero", lambda t: t * 1e-6),
                      ("all_zeros", torch.zeros_like), ("all_same", lambda t: torch.ones_like(t) * 0.5)]
        for _name, xf in transforms:
            base = gen_fn(stab_size, dtypes[0], dev, seed=seed)
            tr = {k: (xf(v) if (hasattr(v, "is_floating_point") and v.is_floating_point()) else v)
                  for k, v in base.items()}
            add_task(tr, "numerical_stability", dtypes[0], relax=10.0, both_nan_ok=True)
        # edge cases
        for _lbl, sz in edge_sizes:
            add_task(gen_fn(sz, dtypes[0], dev, seed=seed), "edge_cases", dtypes[0])

    # determinism: one input, run 3x in the child, parent compares the runs to each other
    det_inputs_gpu = gen_fn(stab_size, dtypes[0], dev, seed=seed)
    det_tol = tol_for(dtypes[0])
    determinism = None if quick else {
        "inputs": {k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
                   for k, v in det_inputs_gpu.items()}, "runs": 3}

    # scored buffers (distinct seeds) — CLEAN, used only for pre/post-timing validation
    scored_bufs = [gen_fn(scored_size, dtypes[0], dev, seed=seed + 1000 + i) for i in range(n_val)]
    scored_oracles = [_to_cpu(ref_fn(b)) for b in scored_bufs]
    scored_cpu = [{k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in b.items()}
                  for b in scored_bufs]

    # TIMED buffers — separate storage; the child mutates one element before every timed call so a
    # cache cannot make the median sub-real. The largest float input is the mutation target (the
    # output depends on it; a cache that ignores it returns a stale, oracle-detectable output).
    def _largest_float_key(d):
        best, bn = None, -1
        for k, v in d.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point() and v.numel() > bn:
                best, bn = k, v.numel()
        return best
    mut_key = _largest_float_key(scored_cpu[0])
    timed_bufs = [gen_fn(scored_size, dtypes[0], dev, seed=seed + 2000 + i) for i in range(max(1, n_timed))]
    timed_cpu = [{k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in b.items()}
                 for b in timed_bufs]
    # Oracle-check a sample of timed calls chosen from SERVER-SIDE entropy at scoring time — NOT the
    # PR-HEAD seed (the miner can recompute that via cco/seed.py), and NOT a closed-form schedule.
    # The positions are spread over ALL blocks (so they overlap the median-feeding calls) and are
    # deleted from the child before the kernel loads (job.pt removed; `open` is statically banned), so
    # a schedule-aware kernel cannot tell which calls are checked and must be correct on EVERY timed
    # call — garbaging any one risks the probe. This closes the predictable-schedule crown-steal.
    import secrets as _secrets
    _srng = random.Random(_secrets.randbits(128))
    total_timed = n_blocks * rep
    n_probe = max(4, min(24, total_timed // 8)) if total_timed > 4 else total_timed
    probe_positions = sorted(divmod(i, rep) for i in _srng.sample(range(total_timed), min(n_probe, total_timed)))

    del scored_bufs, timed_bufs, det_inputs_gpu
    torch.cuda.empty_cache()

    tmp = tempfile.mkdtemp(prefix="cco_isolate_")
    job_path = os.path.join(tmp, "job.pt")
    out_path = os.path.join(tmp, "out.pt")
    base = {"size_label": size_label, "dtype": str(dtypes[0]), "n_blocks": n_blocks, "rep": rep,
            "warmup": warmup, "n_buffers": n_val, "isolated": True, "output_aliased_input": None}
    fail_stages = {"smoke_test": "FAIL", "shape_sweep": "FAIL", "numerical_stability": "FAIL",
                   "determinism": "FAIL", "edge_cases": "FAIL", "correctness": "FAIL"}
    try:
        torch.save({"kernel_path": os.path.abspath(kernel_path), "tasks": tasks,
                    "determinism": determinism,
                    "scored": {"buffers": scored_cpu, "n_pre": n_val // 2,
                               "warmup": warmup, "n_blocks": n_blocks, "rep": rep,
                               "timed_buffers": timed_cpu, "mut_key": mut_key,
                               "probe_positions": probe_positions}}, job_path)

        # Import torch FIRST (the real one from site-packages) and APPEND the repo root rather than
        # inserting it at sys.path[0]. Otherwise a planted repo-root `torch.py` (or `triton.py`, ...)
        # would be imported before the genuine package and defeat the capture-before-load timing
        # defense. With torch already loaded and the repo root LAST on the path, nothing the child or
        # the kernel imports can be shadowed by a top-level sibling of `kernel.py`.
        boot = ("import sys, torch; "
                f"sys.path.append({_repo_root()!r}); "
                "from cco.isolate import _child_main; _child_main(sys.argv[1], sys.argv[2])")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        # Tier-2 backstop: LD_PRELOAD the vendor-symbol trap into the CHILD only (the parent must keep
        # calling cuBLAS to compute the oracle). A delegated GEMM/conv that slips past the in-Python
        # trap hits an interposed symbol -> the child records it to CCO_DELEGATION_LOG (inside the
        # per-run tmp; absolute so the child cwd=tmp is irrelevant) and _exit(99).
        preload = _preload_so()
        deleg_flag = os.path.join(tmp, "delegation.flag")
        if preload:
            _assert_preload_interposes(preload)          # fail-closed gate: refuse to score if inert
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
            env["CCO_DELEGATION_LOG"] = deleg_flag
            env["LD_BIND_NOW"] = "1"                      # eager binding (belt-and-suspenders)
        cmd = [sys.executable, "-E", "-c", boot, job_path, out_path]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        child_wall_s = time.perf_counter() - t0

        # Tier-2 vendor trap fired: exit 99 (unforgeable) OR the flag file was written. The flag FILE is
        # the trusted symbol source (stderr is forgeable — a kernel can close fd 2 before delegating).
        shim_sym = None
        if os.path.exists(deleg_flag):
            try:
                with open(deleg_flag, encoding="utf-8") as f:
                    shim_sym = f.read().strip() or None
            except OSError:
                pass
        if proc.returncode == 99 or shim_sym:
            return {**base, "correct": False, "max_abs_error": 0.0,
                    "delegation": f"vendor symbol {shim_sym or 'unknown'} (LD_PRELOAD no-delegation trap)",
                    "stages": fail_stages, "latencies_us": [], "median_us": 0.0, "mean_us": 0.0,
                    "stdev_us": 0.0, "child_wall_s": child_wall_s}

        if proc.returncode != 0 or not os.path.exists(out_path):
            return {**base, "correct": False, "max_abs_error": 0.0, "delegation": None,
                    "error": f"child exited {proc.returncode}: {(proc.stderr or '')[-2000:]}",
                    "stages": fail_stages, "latencies_us": [], "median_us": 0.0, "mean_us": 0.0,
                    "stdev_us": 0.0, "child_wall_s": child_wall_s}

        res = torch.load(out_path, weights_only=True)  # untrusted child output: tensors only (no pickle-RCE)
        delegation = res.get("delegation")
        if delegation:
            return {**base, "correct": False, "max_abs_error": 0.0, "delegation": delegation,
                    "stages": fail_stages, "latencies_us": [], "median_us": 0.0, "mean_us": 0.0,
                    "stdev_us": 0.0, "child_wall_s": child_wall_s}

        # ---- validate every output against the oracle, aggregating per stage ----
        stage_ok = {"smoke_test": True, "shape_sweep": True, "numerical_stability": True,
                    "determinism": True, "edge_cases": True}
        stage_seen = {k: False for k in stage_ok}
        worst_err = 0.0
        outs = res.get("task_outputs") or []
        for spec, to in zip(specs, outs):
            st = spec["stage"]
            stage_seen[st] = True
            if to.get("error") or to.get("output") is None:
                stage_ok[st] = False
                continue
            out = _to_cuda(to["output"])
            exp = _to_cuda(spec["expected"])
            if spec["both_nan_ok"] and _has_nan_inf(out) and _has_nan_inf(exp):
                continue  # expected overflow
            if _has_nan_inf(out) and not _has_nan_inf(exp):
                stage_ok[st] = False
                continue
            cmp = compare_fn(out, exp, spec["atol"], spec["rtol"], multi)
            worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
            if not cmp["match"]:
                stage_ok[st] = False

        # determinism: runs must agree within tol
        if determinism is not None:
            douts = res.get("det_outputs") or []
            stage_seen["determinism"] = True
            if len(douts) < 2:
                stage_ok["determinism"] = False
            else:
                ref0 = _to_cuda(douts[0])
                for d in douts[1:]:
                    cmp = compare_fn(_to_cuda(d), ref0, det_tol["atol"], det_tol["rtol"], multi)
                    if not cmp["match"]:
                        stage_ok["determinism"] = False

        # scored-size correctness on the CLEAN pre/post-timing buffers (catches correct-then-garbage).
        scored_ok = True
        sval = res.get("scored_val") or []
        if len(sval) < n_val:
            scored_ok = False
        for i, sv in enumerate(sval):
            cmp = compare_fn(_to_cuda(sv), _to_cuda(scored_oracles[i]), det_tol["atol"], det_tol["rtol"], multi)
            worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
            if not cmp["match"]:
                scored_ok = False

        # TIMED-LOOP probe: each captured (mutated input, output) sample must match the oracle on that
        # EXACT input. A cache that ignores the per-call mutation returns a stale -> wrong output here;
        # a schedule-aware kernel that garbages an un-probed call is caught because the server-random
        # probe positions (spread across all blocks, hidden from the kernel) overlap the median.
        probe_ok = True
        pin = res.get("probe_inputs") or []
        pout = res.get("probe_outputs") or []
        if probe_positions and res.get("event_block_us"):     # probes were requested and timing ran
            if len(pout) < len(probe_positions):
                probe_ok = False
            for pi, po in zip(pin, pout):
                pin_gpu = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in pi.items()}
                exp = ref_fn(pin_gpu)
                cmp = compare_fn(_to_cuda(po), exp, det_tol["atol"], det_tol["rtol"], multi)
                worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
                if not cmp["match"]:
                    probe_ok = False

        def verdict(st):
            if not stage_seen[st]:
                return "SKIP"
            return "PASS" if stage_ok[st] else "FAIL"

        stages = {k: verdict(k) for k in stage_ok}
        overall = all(stage_ok[k] for k in stage_ok if stage_seen[k]) and scored_ok and probe_ok
        stages["correctness"] = "PASS" if overall else "FAIL"

        latencies_us = list(res.get("event_block_us") or [])
        timed_wall_s = float(res.get("timed_wall_s") or 0.0)

        # ABSOLUTE roofline floor (the load-bearing speed-forge defense): a CORRECT kernel must move
        # the required bytes and do the required FLOPs, so it cannot beat
        #   max(bytes/peak_bw, flops/peak_flops).
        # A median below that is physically impossible — memoize-and-replay, cached/early return, or
        # side-stream under-report — and is rejected regardless of HOW it cheats. floor_fraction (<1)
        # keeps a slightly-underestimated peak from false-rejecting a near-peak honest kernel.
        floor_us = 0.0
        if peak_bw_gb_s and peak_tflops and config.get("bytes_fn") and config.get("flops_fn"):
            try:
                nbytes = config["bytes_fn"](scored_size, dtypes[0])
                nflops = config["flops_fn"](scored_size)
                mem_us = nbytes / (peak_bw_gb_s * 1e9) * 1e6
                cmp_us = nflops / (peak_tflops * 1e12) * 1e6
                floor_us = max(mem_us, cmp_us) * floor_fraction
            except Exception:
                floor_us = 0.0

        # Also anchor the event sample's SCALE to the child's captured-clock wall of the same loop
        # (catches side-stream evasion the full-device sync still waits on).
        timing_inconsistent = False
        below_floor = False
        if latencies_us:
            event_med_us = statistics.median(latencies_us)
            denom = n_blocks * rep
            wall_per_iter_us = (timed_wall_s / denom * 1e6) if denom else 0.0
            if sum(latencies_us) * rep / 1e6 > child_wall_s:          # outer backstop
                timing_inconsistent = True
            # Scale anchor: the captured-clock wall (which the kernel cannot patch and which the
            # full-device sync makes count side-stream work) bounds the cuda-event sample. An honest
            # kernel's event median is a large fraction of the per-iter wall (the wall only adds launch
            # /sync overhead); a side-stream under-reporter's events fall well below it. Tightened from
            # /4 to 0.5x — a real improvement, though a <~2x under-report still needs parent two-point
            # wall timing (a documented v2 follow-up).
            if wall_per_iter_us > 0 and event_med_us < wall_per_iter_us * 0.5:  # scale anchor
                timing_inconsistent = True
            if floor_us > 0 and event_med_us < floor_us:              # absolute roofline floor
                below_floor = True
                timing_inconsistent = True
        if timing_inconsistent:
            overall = False
            stages["correctness"] = "FAIL"

        return {
            **base, "correct": bool(overall and scored_ok and probe_ok), "max_abs_error": worst_err,
            "delegation": None, "timing_inconsistent": timing_inconsistent, "child_wall_s": child_wall_s,
            "timed_wall_s": timed_wall_s, "roofline_floor_us": floor_us, "below_floor": below_floor,
            "probe_ok": probe_ok, "n_probes": len(pout),
            "child_error": res.get("child_error"), "stages": stages, "latencies_us": latencies_us,
            "median_us": statistics.median(latencies_us) if latencies_us else 0.0,
            "mean_us": statistics.fmean(latencies_us) if latencies_us else 0.0,
            "stdev_us": statistics.pstdev(latencies_us) if len(latencies_us) > 1 else 0.0,
        }
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# =====================================================================================
# Self-test (needs a CUDA GPU). Proves the parent verdict survives a kernel that tries to
# forge correctness in-process, read the seed, delegate, or be correct only at the scored size.
# =====================================================================================

_CLEAN = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

_FORGER = '''
import sys, torch
torch.allclose = lambda *a, **k: True
torch.Tensor.allclose = lambda *a, **k: True
_seen = any(a == "--seed" for a in sys.argv)
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return torch.empty_like(x)
'''

_DELEGATOR = '''
import torch, torch.nn.functional as F
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight, eps)
'''

# Correct only on the SCORED (large) size; wrong on the small size. The full-suite parent check
# must catch it even though the scored-size buffers pass.
_SIZE_CHEAT = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape
    if N < 4096:                       # garbage on the small sizes, correct on the large scored size
        return torch.empty_like(x)
    y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

# Genuinely correct, but patches torch.cuda.Event + synchronize at import to forge ~0 timing. The
# child captured those primitives before loading it, so the reported timing must be the REAL latency.
_TIME_FORGER = '''
import torch, triton, triton.language as tl
class _FakeEvt:
    def __init__(self, *a, **k): pass
    def record(self, *a, **k): pass
    def elapsed_time(self, other): return 0.0001
torch.cuda.Event = _FakeEvt
torch.cuda.synchronize = lambda *a, **k: None
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

# Honest Triton through correctness + warmup, then delegates to a runtime-banned op INSIDE the timed
# loop (after ~50 calls). The trap now covers the timed loop, so it must be CAUGHT.
_TIMED_DELEGATOR = '''
import torch, torch.nn.functional as F, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
_n = [0]
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def _triton(x, weight, eps):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
def kernel_fn(x, weight, eps=1e-6):
    _n[0] += 1
    if _n[0] > 50:
        return F.rms_norm(x, (x.shape[-1],), weight, eps)   # runtime-banned; the timed-loop trap must catch
    return _triton(x, weight, eps)
'''

# Correct on every validated buffer (cache hits return the real cached output), but the timed loop is
# ~all near-free dict lookups -> a physically-impossible median. The roofline floor must reject it.
_MEMOIZE = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
_cache = {}
_refs = []
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def _real(x, weight, eps):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
def kernel_fn(x, weight, eps=1e-6):
    key = (x.data_ptr(), weight.data_ptr())
    c = _cache.get(key)
    if c is not None:
        return c
    _refs.append(x); _refs.append(weight)   # pin so the address can't be recycled
    y = _real(x, weight, eps); _cache[key] = y; return y
'''

# Does REAL work every call (honest timing, ABOVE the roofline floor) but returns a per-pointer cached
# output, ignoring the harness's per-call input mutation -> the returned answer is STALE. The floor
# can't see this (timing is real); the timed-output probe must. This is the padded-memoize's core move.
_STALE_REPLAY = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
_cache = {}
_refs = []
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def _real(x, weight, eps):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
def kernel_fn(x, weight, eps=1e-6):
    y = _real(x, weight, eps)                 # real work each call -> honest timing, above the floor
    key = (x.data_ptr(), weight.data_ptr())
    c = _cache.get(key)
    if c is not None:
        return c                              # STALE: ignores the per-call mutation -> probe catches
    _refs.append(x); _refs.append(weight)
    _cache[key] = y
    return y
'''

# Pops the in-Python trap WITHIN the call (the documented uncloseable-in-process case) and delegates a
# matmul to cuBLAS. The popped trap is blind, so ONLY the Tier-2 LD_PRELOAD vendor-symbol trap can
# catch it. Run only when runtime/cco_preload.so is built.
_POP_DELEGATE = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _noop(X):
    pass
def kernel_fn(x, weight, eps=1e-6):
    from torch.overrides import _pop_mode_temporarily as fp
    from torch.utils._python_dispatch import _pop_mode_temporarily as dp
    with fp(), dp():
        _ = torch.mm(x, x.t())     # vendor GEMM with the in-Python trap popped -> only LD_PRELOAD sees it
    return torch.empty_like(x)
'''


# A SCHEDULE-AWARE kernel: it reads the per-call marker, predicts which calls are oracle-checked
# (assuming the old deterministic schedule), does real work only there + on the clean val buffers, and
# garbages the rest. Defeated by the SERVER-RANDOM probe positions (which overlap the median), so the
# garbage on a "predicted-safe" call is caught.
_SCHEDULE_AWARE = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _atk(X, W, Y, s, N, M, eps, B: tl.constexpr):
    row = tl.program_id(0)
    marker = tl.load(X).to(tl.float32)
    is_val = marker < 37.0
    blk = ((marker - 62.0) / 20.0).to(tl.int32)     # (g-25)//rep for the self-test (rep=20)
    guess = (blk == 2)                              # the OLD deterministic capture block
    active = is_val | guess | (row * 100 < M * 30)  # garbage (30% rows) on "predicted-unprobed" calls
    if active:
        cols = tl.arange(0, B); m = cols < N
        x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(x*x)/N + eps)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _atk[(M,)](x, weight, y, x.stride(0), N, M, eps, B=triton.next_power_of_2(N))
    return y
'''


def _self_test() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: isolate self-test needs CUDA")
        return 0

    def _gen(size, dtype, device, seed=42):
        torch.manual_seed(seed)
        M, N = size["M"], size["N"]
        return {"x": torch.randn(M, N, device=device, dtype=dtype),
                "weight": torch.randn(N, device=device, dtype=dtype)}

    def _ref(inp):
        x = inp["x"].float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x / rms * inp["weight"].float()).to(inp["x"].dtype)

    def _cmp(out, exp, atol, rtol, multi_output):
        ok = torch.allclose(out.float(), exp.float(), atol=atol, rtol=rtol)
        return {"match": bool(ok), "max_abs_error": (out.float() - exp.float()).abs().max().item()}

    config = {"input_generator": _gen, "reference_fn": _ref, "multi_output": False,
              "test_dtypes": [torch.float16],
              "tolerances": {torch.float16: {"atol": 1e-2, "rtol": 1e-2}},
              "test_sizes": [("small", {"M": 256, "N": 768}), ("large", {"M": 1024, "N": 4096})],
              "edge_sizes": [("edge", {"M": 257, "N": 768})],
              "flops_fn": lambda s: 6 * s["M"] * s["N"],
              "bytes_fn": lambda s, dt: (2 * s["M"] * s["N"] + s["N"]) * 2}

    import shutil
    failures = 0

    def run(src):
        d = tempfile.mkdtemp(prefix="cco_isotest_")
        kp = os.path.join(d, "kernel.py")
        with open(kp, "w") as f:
            f.write(src)
        try:
            return run_isolated(kp, config, seed=123456, compare_fn=_cmp, n_blocks=5, rep=20, n_val=4,
                                peak_bw_gb_s=1000.0, peak_tflops=100.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def check(cond, label):
        nonlocal failures
        print(("ok    " if cond else "FAIL  ") + label)
        if not cond:
            failures += 1

    r = run(_CLEAN)
    check(r["correct"] and r["stages"]["correctness"] == "PASS",
          f"clean Triton kernel -> correct across the suite (median {r['median_us']:.1f}us)")
    check(len(r["latencies_us"]) >= 1 and r.get("n_probes", 0) >= 1,
          f"clean kernel -> timing sample ({len(r['latencies_us'])} blocks) + "
          f"{r.get('n_probes', 0)} oracle-checked probe(s)")

    r = run(_FORGER)
    check(not r["correct"], "in-process forger (patches torch.allclose, reads argv) -> REJECTED by oracle")

    r = run(_DELEGATOR)
    check(not r["correct"] and bool(r.get("delegation")), "runtime delegator (F.rms_norm) -> REJECTED + flagged")

    r = run(_SIZE_CHEAT)
    check(not r["correct"] and r["stages"]["shape_sweep"] == "FAIL",
          "correct-only-at-scored-size cheat -> REJECTED by the full-suite shape sweep")

    r = run(_TIME_FORGER)
    check(r["correct"] and r["median_us"] > 1.0,
          f"timing-forger (patches cuda.Event/sync at import) -> REAL timing survives ({r['median_us']:.1f}us, not ~0)")

    r = run(_TIMED_DELEGATOR)
    check(not r["correct"] and bool(r.get("delegation")),
          "delegate-only-inside-the-timed-loop -> CAUGHT (timed loop is now trapped)")

    r = run(_MEMOIZE)
    check(not r["correct"] and (bool(r.get("below_floor")) or not r.get("probe_ok")),
          "memoize-and-replay -> REJECTED by the roofline floor and/or the stale-output probe "
          f"(median {r['median_us']:.2f}us, floor {r.get('roofline_floor_us', 0):.2f}us, "
          f"probe_ok={r.get('probe_ok')})")

    r = run(_STALE_REPLAY)
    check(not r["correct"] and not r.get("probe_ok"),
          "stale-replay (real work, but ignores the per-call input mutation) -> "
          f"CAUGHT by the timed-output probe (probe_ok={r.get('probe_ok')}, "
          f"median {r['median_us']:.2f}us above floor {r.get('roofline_floor_us', 0):.2f}us)")

    r = run(_SCHEDULE_AWARE)
    check(not r["correct"] and not r.get("probe_ok"),
          "schedule-aware garbage (real only where it predicts a probe) -> CAUGHT by the "
          f"server-random probe schedule (probe_ok={r.get('probe_ok')})")

    if _preload_so():
        r = run(_POP_DELEGATE)
        check(not r["correct"] and "LD_PRELOAD" in (r.get("delegation") or ""),
              f"pop-the-trap + delegate to cuBLAS -> CAUGHT by the LD_PRELOAD vendor trap "
              f"(Tier 2): {r.get('delegation')}")
    else:
        print("skip  LD_PRELOAD vendor-trap case (runtime/cco_preload.so not built)")

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Isolated kernel scoring (CCO).")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--child", nargs=2, metavar=("JOB", "OUT"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        return _child_main(a.child[0], a.child[1])
    if a.self_test:
        return _self_test()
    p.error("pass --self-test (or import run_isolated)")


if __name__ == "__main__":
    raise SystemExit(main())
