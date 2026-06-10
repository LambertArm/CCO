"""
cco/manifest_tool.py — Gate-2 file-integrity for CCO competition submissions (Step 3).

Gate 2 pins every LOCKED file so that, at a PR HEAD, only the single
mutable artifact (`kernel.py`) may differ from the canonical tree. This module generates and
verifies `manifest.json`.

The important property — which closes a real RCE the red-team found — is that verification
pins the **full directory listing**, not just the hashes of known files: `kernel_configs/`
auto-imports any `*.py` present, so a miner who *adds* an unlisted `kernel_configs/evil.py`
would inject code into the scoring process. So verify() flags THREE failure modes for every
locked path:
  * modified — a listed file whose sha256 differs from canonical,
  * missing  — a listed file absent at the PR HEAD,
  * unlisted — a file present under a locked path but NOT in the manifest (the injection case).

Only the artifact (`kernel.py`) is excluded from the manifest and allowed to differ.

`manifest.json` is authoritative on `main` (manifest authority lives on main): verify() reads
the locked-path set and hashes FROM the manifest, not from the PR's copy, so a miner editing
their own manifest changes nothing. (Complementary check in the gate pipeline's Gate 2: assert
the PR's git diff touches ONLY `kernel.py` — that also catches a stray top-level file like
`sitecustomize.py` outside any locked path.)

Usage:
    uv run --no-project python cco/manifest_tool.py --self-test
    uv run --no-project python cco/manifest_tool.py generate [--root .] [--out manifest.json] [--paths P ...]
    uv run --no-project python cco/manifest_tool.py verify   [--root .] [--manifest manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

MANIFEST_VERSION = 1
DEFAULT_ARTIFACT = "kernel.py"

# Competition locked roots (files + dirs). The canonical set is cco.config.json's locked_paths —
# `generate` reads it automatically when present at --root; this fallback mirrors it and must be
# kept in sync. Missing entries are skipped at generate time, with a WARNING (an unpinned path is
# an unprotected path).
DEFAULT_LOCKED_PATHS = [
    "benchmark.py", "references", "kernel_configs", "cco", "champions",
    "runtime", "cco.config.json", "payload-schema.json", "pyproject.toml", "uv.lock",
]

IGNORE_DIRNAMES = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
IGNORE_SUFFIXES = (".pyc", ".pyo", ".pyd")
IGNORE_BASENAMES = {".DS_Store", ".gitkeep"}


def _is_ignored(basename: str) -> bool:
    return (basename in IGNORE_BASENAMES
            or any(basename.endswith(s) for s in IGNORE_SUFFIXES))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_locked_files(root: str, locked_paths, artifact: str):
    """Yield (relpath_posix, abspath) for every non-ignored file under the locked paths."""
    for lp in locked_paths:
        ap = os.path.join(root, lp)
        if not os.path.exists(ap):
            continue
        if os.path.isfile(ap):
            base = os.path.basename(ap)
            if not _is_ignored(base):
                rel = lp.replace("\\", "/")
                if rel != artifact:
                    yield rel, ap
            continue
        for dirpath, dirnames, filenames in os.walk(ap):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRNAMES)
            for fn in sorted(filenames):
                if _is_ignored(fn):
                    continue
                abspath = os.path.join(dirpath, fn)
                rel = os.path.relpath(abspath, root).replace("\\", "/")
                if rel != artifact:
                    yield rel, abspath


def compute_manifest(root: str, locked_paths=None, artifact: str = DEFAULT_ARTIFACT) -> dict:
    locked_paths = list(locked_paths if locked_paths is not None else DEFAULT_LOCKED_PATHS)
    files = {rel: _sha256_file(ap) for rel, ap in _iter_locked_files(root, locked_paths, artifact)}
    present = [p for p in locked_paths if os.path.exists(os.path.join(root, p))]
    return {
        "version": MANIFEST_VERSION,
        "artifact": artifact,
        "locked_paths": present,
        "files": dict(sorted(files.items())),
    }


def verify(root: str, manifest: dict, artifact: str | None = None) -> list[tuple[str, str, str]]:
    """Return a list of (kind, relpath, detail) violations; empty == the tree matches canonical.

    kind is one of: modified | missing | unlisted.
    """
    artifact = artifact or manifest.get("artifact", DEFAULT_ARTIFACT)
    locked_paths = manifest["locked_paths"]
    expected = manifest["files"]

    actual = {rel: _sha256_file(ap) for rel, ap in _iter_locked_files(root, locked_paths, artifact)}

    violations: list[tuple[str, str, str]] = []
    for rel, want in expected.items():
        if rel not in actual:
            violations.append(("missing", rel, "file in manifest not found at PR HEAD"))
        elif actual[rel] != want:
            violations.append(("modified", rel, f"sha256 differs (got {actual[rel][:12]}…)"))
    for rel in actual:
        if rel not in expected:
            violations.append(("unlisted", rel, "file present under a locked path but not in manifest"))
    violations.sort(key=lambda v: (v[0], v[1]))
    return violations


# --------------------------------------------------------------------------------------
# Self-test (pure Python; uses a temp tree)
# --------------------------------------------------------------------------------------

def _write(root: str, rel: str, content: str):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _self_test() -> int:
    import shutil
    import tempfile

    failures = 0

    def check(cond: bool, label: str, detail=""):
        nonlocal failures
        if cond:
            print(f"ok    {label}")
        else:
            failures += 1
            print(f"FAIL  {label}  {detail}")

    tmp = tempfile.mkdtemp(prefix="cco_manifest_")
    try:
        # Build a fake locked tree + a mutable artifact.
        a_orig = "def k(): return 1\n"
        b_orig = "X = 2\n"
        _write(tmp, "locked/a.py", a_orig)
        _write(tmp, "locked/sub/b.py", b_orig)
        _write(tmp, "locked/__pycache__/a.cpython-312.pyc", "junk")  # must be ignored
        _write(tmp, "cco.config.json", "{}\n")
        _write(tmp, "kernel.py", "print('artifact v1')\n")
        locked = ["locked", "cco.config.json"]

        man = compute_manifest(tmp, locked, artifact="kernel.py")
        check("locked/a.py" in man["files"] and "locked/sub/b.py" in man["files"],
              "manifest lists locked files")
        check(all("__pycache__" not in p for p in man["files"]),
              "manifest ignores __pycache__/*.pyc")
        check(verify(tmp, man) == [], "clean tree verifies with 0 violations")

        # a) modify a locked file -> 'modified'
        _write(tmp, "locked/a.py", "def k(): return 999\n")
        v = verify(tmp, man)
        check(any(k == "modified" and r == "locked/a.py" for k, r, _ in v), "modified file caught")
        _write(tmp, "locked/a.py", a_orig)

        # b) add an UNLISTED file under a locked path -> 'unlisted' (the RCE case)
        _write(tmp, "locked/evil.py", "import os  # injected\n")
        v = verify(tmp, man)
        check(any(k == "unlisted" and r == "locked/evil.py" for k, r, _ in v), "unlisted injected file caught")
        os.remove(os.path.join(tmp, "locked", "evil.py"))

        # c) remove a locked file -> 'missing'
        os.remove(os.path.join(tmp, "locked", "sub", "b.py"))
        v = verify(tmp, man)
        check(any(k == "missing" and r == "locked/sub/b.py" for k, r, _ in v), "removed file caught")
        _write(tmp, "locked/sub/b.py", b_orig)

        # d) modify the ARTIFACT -> NO violation (kernel.py is the one file allowed to differ)
        _write(tmp, "kernel.py", "print('artifact v2 - a winning submission')\n")
        check(verify(tmp, man) == [], "artifact (kernel.py) may differ -> still 0 violations")

        # restored tree verifies clean again
        check(verify(tmp, man) == [], "restored tree verifies clean")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gate-2 manifest generate/verify for CCO submissions.")
    p.add_argument("--self-test", action="store_true", help="run built-in test cases and exit")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="hash locked paths and write manifest.json")
    g.add_argument("--root", default=".")
    g.add_argument("--out", default="manifest.json", help="output path, or - for stdout")
    g.add_argument("--paths", nargs="*", default=None, help="locked paths (default: built-in set)")
    g.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    g.add_argument("--config", help="read locked_paths + artifact from a cco.config.json (authoritative)")

    v = sub.add_parser("verify", help="verify a working tree against a manifest")
    v.add_argument("--root", default=".")
    v.add_argument("--manifest", default="manifest.json")

    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.cmd == "generate":
        paths, artifact = args.paths, args.artifact
        config_path = args.config
        if config_path is None and paths is None:
            _auto = os.path.join(args.root, "cco.config.json")
            if os.path.isfile(_auto):  # the config is the single source of truth when present
                config_path = _auto
                print(f"locked_paths: using {_auto}", file=sys.stderr)
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                _cfg = json.load(f)
            paths = _cfg.get("locked_paths", paths)
            artifact = _cfg.get("artifact", artifact)
        _effective = list(paths if paths is not None else DEFAULT_LOCKED_PATHS)
        _missing = [p for p in _effective if not os.path.exists(os.path.join(args.root, p))]
        if _missing:
            print(f"WARNING: locked path(s) not found on disk and NOT pinned: {', '.join(_missing)}",
                  file=sys.stderr)
        man = compute_manifest(args.root, paths, artifact)
        text = json.dumps(man, indent=2, sort_keys=True) + "\n"
        if args.out == "-":
            sys.stdout.write(text)
        else:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        print(f"manifest: {len(man['files'])} files across {len(man['locked_paths'])} locked path(s) "
              f"({', '.join(man['locked_paths'])}) -> {args.out}", file=sys.stderr)
        return 0

    if args.cmd == "verify":
        with open(args.manifest, "r", encoding="utf-8") as f:
            man = json.load(f)
        vios = verify(args.root, man)
        if not vios:
            print("manifest OK: working tree matches canonical (only the artifact may differ)")
            return 0
        print(f"manifest REJECT: {len(vios)} violation(s)")
        for kind, rel, detail in vios:
            print(f"  [{kind}] {rel}  {detail}")
        return 1

    p.error("pass --self-test, or a subcommand: generate | verify")


if __name__ == "__main__":
    raise SystemExit(main())
