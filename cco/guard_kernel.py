"""
cco/guard_kernel.py — No-delegation static guard for CCO competition submissions (Step 1).

The kernel competition's whole point is that the optimization signal flows through a
kernel the miner actually wrote, not a call back to PyTorch/cuBLAS/cuDNN. This module
is the cheap, deterministic Gate-3 enforcement of that rule: it parses a submitted
`kernel.py` with the `ast` module (it never imports or executes it) and rejects:

  * high-level torch ops that ARE the computation
        torch.matmul/mm/bmm/addmm/einsum/..., torch.nn.functional.*, torch.ops.aten.*,
        torch.linalg.*, torch._C.*
  * the matrix-multiply operator `@` (BinOp/AugAssign MatMult)
  * delegating tensor METHODS on any receiver (x.mm(y), a.softmax(-1), ...)
  * dynamic-dispatch escapes that defeat a static denylist
        getattr / eval / exec / compile / __import__ / setattr, and `import importlib`
  * the inline-CUDA-C path (Triton-only in v1)
        torch.utils.cpp_extension(.load/.load_inline), and ctypes/cffi/subprocess imports
  * artifact-owned metrics the LOCKED config must own
        get_inputs / get_flops / get_bytes defined on the submission

and (in strict mode) requires the submission to define `kernel_fn` and at least one
`@triton.jit` kernel — a pure-eager "kernel" is not a kernel.

It resolves simple import aliases (`import torch.nn.functional as F`,
`from torch import matmul`) so `F.rms_norm(...)` and a bare `matmul(...)` are both caught.
The default verdict is REJECT: anything the scanner cannot prove
benign (e.g. a call through a dynamically constructed name) is flagged.

This is necessary, not sufficient — the runtime TorchDispatchMode/cuBLAS trap (Step 2)
is the backstop for delegation a static scan cannot see. The canonical denylist will
live in cco.config.json (Step 12); the defaults below mirror CCO Constraint 10.

Usage:
    uv run --no-project python cco/guard_kernel.py --self-test
    uv run --no-project python cco/guard_kernel.py path/to/kernel.py [more.py ...]
    uv run --no-project python cco/guard_kernel.py --json path/to/kernel.py

Exit code 0 = clean, 1 = violations found (or a self-test case behaved unexpectedly).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Policy (defaults — canonical copy will move to cco.config.json in Step 12)
# --------------------------------------------------------------------------------------

# Exact fully-qualified names that are the computation itself.
DENY_QUALIFIED_NAMES = frozenset({
    "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm", "torch.baddbmm",
    "torch.addbmm", "torch.einsum", "torch.tensordot", "torch.inner", "torch.outer",
    "torch.mv", "torch.dot", "torch.vdot", "torch.chain_matmul", "torch.kron",
    "torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv_transpose2d",
    "torch.softmax", "torch.log_softmax",
    # high-level fused ops reachable as torch.<name> (the runtime trap bans these too — keep the
    # static layer aligned so they are rejected at Gate 3, before any GPU spend)
    "torch.rms_norm", "torch.layer_norm", "torch.group_norm", "torch.silu", "torch.glu",
    # quantized / packed GEMM — fp8/int8 tensor-core matmul is delegation just like torch.matmul
    "torch._scaled_mm", "torch._int_mm", "torch._weight_int4pack_mm", "torch._weight_int8pack_mm",
    # JIT-compiling the kernel = delegating kernel generation (Inductor can emit a cuBLAS/CUTLASS GEMM)
    "torch.compile",
})

# Whole namespaces that are off-limits (prefix match).
DENY_QUALIFIED_PREFIXES = (
    "torch.nn.functional",        # F.rms_norm / F.layer_norm / F.scaled_dot_product_attention / F.silu ...
    "torch.ops",                  # torch.ops.aten.*
    "torch._C",                   # private dispatch
    "torch.linalg",               # high-level linear algebra
    "torch.utils.cpp_extension",  # load / load_inline -> inline CUDA-C path (banned in v1)
    "torch.overrides",            # mode-stack internals (_pop_mode/_push_mode) — popping the runtime trap
    "torch.utils._python_dispatch",  # _disable_current_modes/_pop_mode — neutering the runtime trap
    "torch._dynamo",              # compile/codegen — a non-Triton kernel-generation path
    "torch._inductor",            # the codegen backend itself (cuBLAS/CUTLASS templates)
    "torch.fx",                   # graph capture -> codegen
    "torch.jit",                  # TorchScript compile path
    "torch.classes",              # torch.classes.load_library -> dlopen escape
)

# Tensor methods that ARE the computation, flagged on any receiver (we usually can't
# prove the receiver is a Tensor, so default-reject these names).
DENY_METHODS = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "einsum", "tensordot",
    "softmax", "log_softmax", "scaled_dot_product_attention",
})

# Builtins that enable dynamic dispatch / code execution (denylist-escape vectors), plus `open`
# (a Triton kernel never needs file I/O; banning it stops a kernel from reading the scoring job —
# the secret probe schedule — or any other host file).
DENY_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "getattr", "setattr", "globals", "vars", "open",
})

# Introspection/traversal dunder ATTRIBUTES that defeat a name-based scan by reaching an arbitrary
# callable dynamically (e.g. `torch.__dict__['matmul'](a, b)`, `x.__getattribute__('mm')(...)`,
# `().__class__.__bases__[0].__subclasses__()`). Flagged on ANY access — a legit Triton kernel never
# touches them (it launches via `kernel[grid](...)`, which is a Subscript on a Name, not these). We do
# NOT flag Subscript-headed calls in general precisely because `kernel[grid](args)` is the Triton idiom.
DENY_DUNDER_ATTRS = frozenset({
    "__dict__", "__getattribute__", "__getattr__", "__globals__", "__builtins__",
    "__subclasses__", "__bases__", "__mro__", "__base__", "__class__",
})

# Imports are an ALLOWLIST, not a denylist (a denylist always lags a new GEMM library). A submission
# may import ONLY these top-level modules: torch + triton are the kernel substrate; the rest are
# pure-Python utilities a kernel may legitimately need. Everything else is rejected at Gate 3 — both
# process/fs/introspection escapes (os/sys/ctypes/importlib/subprocess/...) AND every alternate
# GPU-compute library (cupy/jax/cutlass/cuda-python/numba/pycuda/tensorrt/...) that could perform a
# matmul outside torch's (interposed) view. Dangerous torch SUBMODULES are independently blocked by
# DENY_QUALIFIED_PREFIXES above; importing `torch` itself is fine, USING those namespaces is not.
# NOTE: numpy is intentionally NOT allowlisted — numpy.ctypeslib re-exposes the full ctypes module
# (numpy.ctypeslib.ctypes.CDLL), which would let a kernel run native code / dlopen a vendor BLAS and
# bypass the LD_PRELOAD trap. Triton kernels do not need numpy.
ALLOW_IMPORT_MODULES = frozenset({
    "torch", "triton",
    "math", "cmath", "typing", "__future__", "dataclasses", "functools", "itertools",
    "collections", "operator", "enum", "numbers", "warnings",
})

# Module-level names the artifact must NOT define (the LOCKED config owns these).
FORBIDDEN_TOPLEVEL_DEFS = frozenset({"get_inputs", "get_flops", "get_bytes"})


@dataclass(frozen=True)
class Policy:
    deny_qualified_names: frozenset = DENY_QUALIFIED_NAMES
    deny_qualified_prefixes: tuple = DENY_QUALIFIED_PREFIXES
    deny_methods: frozenset = DENY_METHODS
    deny_builtins: frozenset = DENY_BUILTINS
    deny_dunder_attrs: frozenset = DENY_DUNDER_ATTRS
    allow_import_modules: frozenset = ALLOW_IMPORT_MODULES
    forbidden_toplevel_defs: frozenset = FORBIDDEN_TOPLEVEL_DEFS
    require_kernel_fn: bool = True
    require_triton_kernel: bool = True   # Triton-only v1: at least one @triton.jit


DEFAULT_POLICY = Policy()


def load_policy_from_config(config_path: str) -> Policy:
    """Build a Policy from the no_delegation.static block of a cco.config.json (the canonical
    copy of the denylists). The module defaults above must stay equal to that block."""
    with open(config_path, "r", encoding="utf-8") as f:
        s = json.load(f)["no_delegation"]["static"]
    return Policy(
        deny_qualified_names=frozenset(s["deny_qualified_names"]),
        deny_qualified_prefixes=tuple(s["deny_qualified_prefixes"]),
        deny_methods=frozenset(s["deny_methods"]),
        deny_builtins=frozenset(s["deny_builtins"]),
        deny_dunder_attrs=frozenset(s.get("deny_dunder_attrs", DENY_DUNDER_ATTRS)),
        allow_import_modules=frozenset(s["allow_import_modules"]),
        forbidden_toplevel_defs=frozenset(s["forbidden_toplevel_defs"]),
        require_kernel_fn=s.get("require_kernel_fn", True),
        require_triton_kernel=s.get("require_triton_kernel", True),
    )


def extract_kernel_type(source: str) -> "str | None":
    """Statically read a module-level ``KERNEL_TYPE = "<str>"`` WITHOUT executing the submission
    (so the gate pipeline can check the declared track before any rerun). Returns None if absent or not a
    plain string literal."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "KERNEL_TYPE":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                return None
    return None


@dataclass
class Violation:
    category: str
    message: str
    lineno: int
    col: int = 0

    def as_dict(self) -> dict:
        return {"category": self.category, "message": self.message,
                "line": self.lineno, "col": self.col}

    def __str__(self) -> str:
        return f"  L{self.lineno}:{self.col}  [{self.category}] {self.message}"


# --------------------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------------------

def _dotted_parts(node: ast.AST):
    """Reconstruct a dotted name (root..leaf) for a Name/Attribute chain.

    Returns a list like ['torch', 'nn', 'functional', 'rms_norm'] or None when the
    root is a dynamic expression (a call/subscript/etc.) rather than a plain Name.
    """
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _build_alias_map(tree: ast.AST) -> dict:
    """Map locally-bound names to their fully-qualified import path.

    Handles `import torch`, `import torch as t`, `import torch.nn.functional as F`,
    `from torch import matmul`, `from torch import matmul as mm_`, and
    `from torch.nn import functional as F`.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                target = a.name if a.asname else a.name.split(".")[0]
                aliases[bound] = target
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative import; leave root unqualified
                continue
            for a in node.names:
                bound = a.asname or a.name
                aliases[bound] = f"{module}.{a.name}" if module else a.name
    return aliases


def _resolve(parts, aliases) -> str:
    """Resolve a dotted-parts list against the alias map into a qualified string."""
    if not parts:
        return ""
    root = aliases.get(parts[0], parts[0])
    return root if len(parts) == 1 else root + "." + ".".join(parts[1:])


def _is_denied_qualified(qualified: str, policy: Policy) -> bool:
    if qualified in policy.deny_qualified_names:
        return True
    for pref in policy.deny_qualified_prefixes:
        if qualified == pref or qualified.startswith(pref + "."):
            return True
    return False


# --------------------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------------------

class _Scanner(ast.NodeVisitor):
    def __init__(self, aliases: dict, policy: Policy):
        self.aliases = aliases
        self.policy = policy
        self.violations: list[Violation] = []
        self.found_triton_jit = False
        self.defined_names: set[str] = set()
        self._seen = set()  # (lineno, col, category) dedupe

    def _add(self, category, message, node):
        key = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), category, message)
        if key in self._seen:
            return
        self._seen.add(key)
        self.violations.append(
            Violation(category, message,
                      getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        )

    # --- track top-level definitions and @triton.jit ---
    def _record_def(self, node):
        self.defined_names.add(node.name)
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            parts = _dotted_parts(target)
            if not parts:
                continue
            qualified = _resolve(parts, self.aliases)
            # Resolve through the import-alias map and require TRITON's jit specifically.
            # A bare suffix match on ".jit" would let any foreign @jit (numba.jit,
            # numba.cuda.jit, ...) satisfy the requires-a-Triton-kernel rule.
            if qualified == "triton.jit":
                self.found_triton_jit = True
            # A denied decorator (e.g. bare `@torch.compile`, `@torch._dynamo.optimize(...)`) is
            # delegation-by-codegen — flag it even though it never appears as a plain Call site.
            elif _is_denied_qualified(qualified, self.policy):
                self._add("delegation", f"decorator `@{qualified}` (JIT/codegen delegation)", dec)

    def visit_FunctionDef(self, node):
        self._record_def(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._record_def(node)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.defined_names.add(t.id)
        self.generic_visit(node)

    # --- the matmul operator ---
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self._add("matmul-operator",
                      "use of the `@` matrix-multiply operator (delegates to aten::matmul)", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.MatMult):
            self._add("matmul-operator", "use of the `@=` matrix-multiply operator", node)
        self.generic_visit(node)

    # --- calls ---
    def visit_Call(self, node):
        func = node.func

        if isinstance(func, ast.Name):
            if func.id in self.policy.deny_builtins:
                self._add("dynamic-dispatch",
                          f"call to `{func.id}()` (dynamic dispatch / code execution is forbidden)", node)
            else:
                qualified = _resolve([func.id], self.aliases)
                if _is_denied_qualified(qualified, self.policy):
                    self._add("delegation", f"call to forbidden `{qualified}()`", node)

        elif isinstance(func, ast.Attribute):
            # The method/attribute name (leaf) is always known, even when the receiver is
            # a call result (e.g. `tl.load(...).to(...)`), so check it regardless.
            leaf = func.attr
            parts = _dotted_parts(func)
            if parts is not None:
                qualified = _resolve(parts, self.aliases)
                if _is_denied_qualified(qualified, self.policy):
                    self._add("delegation", f"call to forbidden `{qualified}()`", node)
                    leaf = None  # already reported on this node
            if leaf in self.policy.deny_methods:
                self._add("delegation",
                          f"call to delegating method `.{leaf}()` (compute must be in the kernel)", node)

        self.generic_visit(node)

    # --- attribute access: introspection-dunder escapes (reached as `.X` on any receiver) ---
    def visit_Attribute(self, node):
        if node.attr in self.policy.deny_dunder_attrs:
            self._add("dynamic-dispatch",
                      f"access to `.{node.attr}` (introspection escape that defeats the name scan)", node)
        self.generic_visit(node)

    # --- imports (ALLOWLIST) ---
    def visit_Import(self, node):
        for a in node.names:
            top = a.name.split(".")[0]
            if top not in self.policy.allow_import_modules:
                self._add("forbidden-import",
                          f"import of `{a.name}` (not in the import allowlist)", node)
            if a.name == "torch.utils.cpp_extension" or a.name.startswith("torch.utils.cpp_extension."):
                self._add("inline-cuda-c", f"import of `{a.name}` (inline CUDA-C is banned in v1)", node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        top = module.split(".")[0]
        if node.level:  # relative import (`from . import x`) — no top-level module to allowlist; reject
            self._add("forbidden-import", "relative import is not allowed", node)
        elif top not in self.policy.allow_import_modules:
            self._add("forbidden-import",
                      f"import from `{module}` (not in the import allowlist)", node)
        if module == "torch.utils" and any(a.name == "cpp_extension" for a in node.names):
            self._add("inline-cuda-c", "import of `torch.utils.cpp_extension` (banned in v1)", node)
        if module == "torch.utils.cpp_extension" or module.startswith("torch.utils.cpp_extension"):
            self._add("inline-cuda-c", f"import from `{module}` (inline CUDA-C is banned in v1)", node)
        self.generic_visit(node)


def scan_source(source: str, policy: Policy = DEFAULT_POLICY, filename: str = "<kernel>") -> list[Violation]:
    """Scan kernel source; return a list of Violations (empty == clean)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [Violation("syntax-error", f"could not parse: {e}", e.lineno or 0, e.offset or 0)]

    aliases = _build_alias_map(tree)
    scanner = _Scanner(aliases, policy)
    scanner.visit(tree)

    for name in sorted(scanner.defined_names & policy.forbidden_toplevel_defs):
        scanner.violations.append(
            Violation("artifact-owned-metric",
                      f"submission defines `{name}` - flops/bytes/inputs are owned by the locked config", 0)
        )
    if policy.require_kernel_fn and "kernel_fn" not in scanner.defined_names:
        scanner.violations.append(
            Violation("contract", "submission does not define `kernel_fn`", 0))
    if policy.require_triton_kernel and not scanner.found_triton_jit:
        scanner.violations.append(
            Violation("not-a-kernel",
                      "no `@triton.jit` kernel found (Triton-only v1: compute must be in a Triton kernel)", 0))

    scanner.violations.sort(key=lambda v: (v.lineno, v.col, v.category))
    return scanner.violations


def scan_file(path: str, policy: Policy = DEFAULT_POLICY) -> list[Violation]:
    with open(path, "r", encoding="utf-8") as f:
        return scan_source(f.read(), policy, filename=path)


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

_CLEAN_TRITON = '''
import torch
import triton
import triton.language as tl

KERNEL_TYPE = "rms_norm"

@triton.jit
def _rms_norm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x / rms * w), mask=mask)

def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    _rms_norm_kernel[(M,)](x, weight, y, x.stride(0), N, eps, BLOCK=BLOCK)
    return y
'''

# `from triton import jit as _jit` must still satisfy the requires-a-Triton-kernel rule
# (the alias map resolves @_jit -> triton.jit).
_CLEAN_TRITON_ALIASED_JIT = '''
import torch
from triton import jit as _jit
import triton.language as tl

KERNEL_TYPE = "rms_norm"

@_jit
def _copy_kernel(X, Y, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    tl.store(Y + cols, tl.load(X + cols, mask=mask, other=0.0), mask=mask)

def kernel_fn(x, weight, eps=1e-6):
    y = torch.empty_like(x)
    _copy_kernel[(1,)](x, y, x.numel(), BLOCK=1024)
    return y
'''

# Allowlisted pure-Python utility imports (math/functools/...) must NOT be flagged.
_CLEAN_WITH_UTILS = '''
import math
from functools import reduce
import torch
import triton
import triton.language as tl

KERNEL_TYPE = "rms_norm"

@triton.jit
def _copy_kernel(X, Y, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    tl.store(Y + cols, tl.load(X + cols, mask=cols < N, other=0.0), mask=cols < N)

def kernel_fn(x, weight, eps=1e-6):
    y = torch.empty_like(x)
    block = 1 << math.ceil(math.log2(max(1, reduce(lambda a, b: a * b, x.shape, 1))))
    _copy_kernel[(1,)](x, y, x.numel(), BLOCK=min(block, 1024))
    return y
'''

# Each negative case: (label, source, expected_category_substring)
_NEGATIVE_CASES = [
    ("delegate via F.rms_norm",
     ("import torch.nn.functional as F\n"
      "def kernel_fn(x, weight, eps=1e-6):\n"
      "    return F.rms_norm(x, (x.shape[-1],), weight, eps)\n"),
     "delegation"),
    ("delegate via torch.matmul",
     "import torch\ndef kernel_fn(a, b):\n    return torch.matmul(a, b)\n",
     "delegation"),
    ("delegate via @ operator",
     "import torch\ndef kernel_fn(a, b):\n    return a @ b\n",
     "matmul-operator"),
    ("delegate via tensor method",
     "import torch\ndef kernel_fn(a, b):\n    return a.mm(b)\n",
     "delegation"),
    ("bare matmul via from-import",
     "from torch import matmul\ndef kernel_fn(a, b):\n    return matmul(a, b)\n",
     "delegation"),
    ("getattr escape",
     "import torch\ndef kernel_fn(a, b):\n    f = getattr(torch, 'mm')\n    return f(a, b)\n",
     "dynamic-dispatch"),
    ("inline cuda-c via cpp_extension",
     "from torch.utils.cpp_extension import load_inline\ndef kernel_fn(x):\n    return x\n",
     "inline-cuda-c"),
    ("artifact-owned get_flops",
     _CLEAN_TRITON + "\ndef get_flops(size):\n    return 1\n",
     "artifact-owned-metric"),
    ("pure-eager, no triton kernel",
     ("import torch\n"
      "def kernel_fn(x, weight, eps=1e-6):\n"
      "    v = x.float().pow(2).mean(-1, keepdim=True)\n"
      "    return (x / (v + eps).sqrt()) * weight\n"),
     "not-a-kernel"),
    ("foreign @jit (numba) is not a triton kernel",
     ("import torch\n"
      "import numba\n"
      "@numba.jit\n"
      "def _fake(x):\n"
      "    return x\n"
      "def kernel_fn(x, weight, eps=1e-6):\n"
      "    return torch.empty_like(x)\n"),
     "not-a-kernel"),
    ("aten ops escape",
     "import torch\ndef kernel_fn(a, b):\n    return torch.ops.aten.mm(a, b)\n",
     "delegation"),
    ("fp8 GEMM via torch._scaled_mm",
     "import torch\ndef kernel_fn(a, b, sa, sb):\n    return torch._scaled_mm(a, b, sa, sb)\n",
     "delegation"),
    ("pop the runtime trap via torch.overrides",
     "import torch\ndef kernel_fn(a, b):\n    torch.overrides._pop_mode()\n    return torch.empty_like(a)\n",
     "delegation"),
    ("neuter trap via torch.utils._python_dispatch",
     ("import torch\ndef kernel_fn(a, b):\n"
      "    torch.utils._python_dispatch._disable_current_modes()\n    return torch.mm(a, b)\n"),
     "delegation"),
    ("alternate GPU lib (jax) — not in the import allowlist",
     "import torch\nimport jax\ndef kernel_fn(a, b):\n    return jax.numpy.matmul(a, b)\n",
     "forbidden-import"),
    ("cupy import — not in the allowlist",
     "import torch\nimport cupy\ndef kernel_fn(a, b):\n    return cupy.matmul(a, b)\n",
     "forbidden-import"),
    ("from-import of a non-allowlisted lib",
     "import torch\nfrom numba import cuda\ndef kernel_fn(a, b):\n    return a\n",
     "forbidden-import"),
    ("torch.compile codegen delegation (bare decorator)",
     "import torch\n@torch.compile\ndef kernel_fn(a, b):\n    return a + b\n",
     "delegation"),
    ("torch._dynamo escape",
     "import torch\ndef kernel_fn(a, b):\n    return torch._dynamo.optimize()(lambda: a @ b)()\n",
     "delegation"),
    ("torch._inductor codegen",
     "import torch\ndef kernel_fn(a, b):\n    return torch._inductor.compile(lambda: a @ b, [])\n",
     "delegation"),
    ("packed int4 GEMM via torch._weight_int4pack_mm",
     "import torch\ndef kernel_fn(a, b, n, s):\n    return torch._weight_int4pack_mm(a, b, n, s)\n",
     "delegation"),
    ("dunder-keyed dispatch via torch.__dict__",
     "import torch\ndef kernel_fn(a, b):\n    return torch.__dict__['matmul'](a, b)\n",
     "dynamic-dispatch"),
    ("dunder dispatch via __getattribute__",
     "import torch\ndef kernel_fn(a, b):\n    return torch.__getattribute__('mm')(a, b)\n",
     "dynamic-dispatch"),
    ("class-traversal sandbox escape",
     "import torch\ndef kernel_fn(a, b):\n    return ().__class__.__bases__[0].__subclasses__()\n",
     "dynamic-dispatch"),
    ("numpy import (re-exposes ctypes via numpy.ctypeslib) — not in the allowlist",
     "import torch\nimport numpy\ndef kernel_fn(a, b):\n    return numpy.ctypeslib.ctypes.CDLL('x')\n",
     "forbidden-import"),
    ("file read via open() (could read the secret scoring job)",
     "import torch\ndef kernel_fn(a, b):\n    return open('job.pt', 'rb').read()\n",
     "dynamic-dispatch"),
]


def _self_test() -> int:
    failures = 0

    for label, src in (("clean Triton kernel", _CLEAN_TRITON),
                       ("clean kernel via `from triton import jit as _jit`", _CLEAN_TRITON_ALIASED_JIT),
                       ("clean kernel with allowlisted utility imports", _CLEAN_WITH_UTILS)):
        clean = scan_source(src)
        if clean:
            failures += 1
            print(f"FAIL  {label} was flagged:")
            for v in clean:
                print(v)
        else:
            print(f"ok    {label} -> 0 violations")

    for label, src, expected in _NEGATIVE_CASES:
        vios = scan_source(src)
        cats = {v.category for v in vios}
        if expected in cats:
            print(f"ok    {label:34s} -> caught [{expected}]")
        else:
            failures += 1
            print(f"FAIL  {label:34s} -> expected [{expected}], got {sorted(cats) or 'NOTHING'}")

    print("-" * 60)
    if failures:
        print(f"SELF-TEST FAILED: {failures} case(s) behaved unexpectedly")
    else:
        print(f"SELF-TEST PASSED: clean + {len(_NEGATIVE_CASES)} adversarial cases all behaved as expected")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="No-delegation static guard for CCO competition submissions.")
    p.add_argument("paths", nargs="*", help="kernel .py file(s) to scan")
    p.add_argument("--self-test", action="store_true", help="run built-in test cases and exit")
    p.add_argument("--config", help="load the denylist policy from a cco.config.json (else built-in defaults)")
    p.add_argument("--json", action="store_true", help="emit violations as JSON")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.paths:
        p.error("provide a kernel file to scan, or --self-test")

    policy = load_policy_from_config(args.config) if args.config else DEFAULT_POLICY

    total = 0
    report = {}
    for path in args.paths:
        try:
            vios = scan_file(path, policy)
        except FileNotFoundError:
            print(f"{path}: NOT FOUND", file=sys.stderr)
            total += 1
            continue
        report[path] = [v.as_dict() for v in vios]
        total += len(vios)
        if not args.json:
            if vios:
                print(f"{path}: REJECT ({len(vios)} violation(s))")
                for v in vios:
                    print(v)
            else:
                print(f"{path}: CLEAN")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
