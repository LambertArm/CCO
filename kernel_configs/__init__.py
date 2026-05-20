"""Auto-discovery registry for kernel benchmark configurations.

Scans this directory for ``*.toml`` files.  For each file the registry:
1. Parses the TOML data (sizes, dtypes, tolerances, edge_sizes, meta flags).
2. Imports the companion ``.py`` module with the same stem name to obtain the
   four required callables: ``input_generator``, ``reference_fn``, ``flops_fn``,
   ``bytes_fn``.
3. Merges everything into a single config dict whose schema matches what
   ``tools/bench.py`` expects in ``KERNEL_CONFIGS``.

Importing this module is cheap: it parses TOML, validates dtype names and
companion-file existence, and AST-checks the companion modules, but does
**not** import ``torch`` or the companion modules themselves.  Heavy work is
deferred until ``KERNEL_CONFIGS`` is first accessed, so the registry can be
imported in environments without a GPU stack (e.g. CI lint runners).
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from ._utils import DTYPE_NAMES

_PKG_DIR = pathlib.Path(__file__).resolve().parent

_REQUIRED_TOP_LEVEL_KEYS = ("test_sizes", "tolerances")


def _load_toml(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _validate_toml(toml_path: pathlib.Path, data: dict) -> None:
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise ValueError(
                f"{toml_path.name}: missing required top-level key '{key}'"
            )

    for name in data["tolerances"]:
        if name not in DTYPE_NAMES:
            raise ValueError(
                f"{toml_path.name}: unknown dtype '{name}' in tolerances. "
                f"Valid: {sorted(DTYPE_NAMES)}"
            )

    test_dtypes = data.get("meta", {}).get("test_dtypes")
    if isinstance(test_dtypes, list):
        for name in test_dtypes:
            if name not in DTYPE_NAMES:
                raise ValueError(
                    f"{toml_path.name}: unknown dtype '{name}' in meta.test_dtypes. "
                    f"Valid: {sorted(DTYPE_NAMES)}"
                )


def _ast_check(py_path: pathlib.Path) -> None:
    try:
        ast.parse(py_path.read_text(), filename=str(py_path))
    except SyntaxError as e:
        raise ValueError(f"{py_path.name}: syntax error - {e}") from e


def _discover_paths() -> Dict[str, pathlib.Path]:
    """Validate registry without importing torch or companion modules.

    Catches the structural failures contributors are most likely to introduce:
    a ``.toml`` with no companion ``.py``, malformed TOML, unknown dtype names,
    or a syntactically broken companion module.
    """
    discovered: Dict[str, pathlib.Path] = {}
    for toml_path in sorted(_PKG_DIR.glob("*.toml")):
        kernel_name = toml_path.stem
        py_path = toml_path.with_suffix(".py")
        if not py_path.exists():
            raise FileNotFoundError(
                f"Kernel config '{toml_path.name}' has no companion "
                f"'{py_path.name}' providing callables"
            )

        data = _load_toml(toml_path)
        _validate_toml(toml_path, data)
        _ast_check(py_path)

        discovered[kernel_name] = toml_path
    return discovered


_DISCOVERED_PATHS: Dict[str, pathlib.Path] = _discover_paths()


def _parse_sizes(raw: list[dict]) -> list[tuple[str, dict]]:
    return [(entry["label"], entry["params"]) for entry in raw]


def _parse_dtypes(raw: list[str]) -> list[Any]:
    from ._utils import DTYPE_MAP

    return [DTYPE_MAP[name] for name in raw]


def _parse_tolerances(raw: dict) -> dict:
    from ._utils import DTYPE_MAP

    out: dict[Any, dict[str, float]] = {}
    for name, tol in raw.items():
        out[DTYPE_MAP[name]] = {"atol": float(tol["atol"]), "rtol": float(tol["rtol"])}
    return out


def _build_config(toml_path: pathlib.Path) -> Dict[str, Any]:
    data = _load_toml(toml_path)
    stem = toml_path.stem

    mod = importlib.import_module(f"kernel_configs.{stem}")

    cfg: Dict[str, Any] = {}

    meta = data.get("meta", {})
    if meta.get("multi_output", False):
        cfg["multi_output"] = True

    cfg["test_sizes"] = _parse_sizes(data["test_sizes"])
    cfg["test_dtypes"] = _parse_dtypes(data["test_dtypes"])
    cfg["tolerances"] = _parse_tolerances(data["tolerances"])

    edge_raw = data.get("edge_sizes", [])
    cfg["edge_sizes"] = _parse_sizes(edge_raw) if edge_raw else []

    cfg["input_generator"] = mod.input_generator
    cfg["reference_fn"] = mod.reference_fn
    cfg["flops_fn"] = mod.flops_fn
    cfg["bytes_fn"] = mod.bytes_fn

    return cfg


_KERNEL_CONFIGS_CACHE: Dict[str, Dict[str, Any]] | None = None


def _build_all_configs() -> Dict[str, Dict[str, Any]]:
    return {
        name: _build_config(path) for name, path in _DISCOVERED_PATHS.items()
    }


def __getattr__(name: str) -> Any:
    # PEP 562: fires on both ``kernel_configs.KERNEL_CONFIGS`` attribute
    # access and ``from kernel_configs import KERNEL_CONFIGS``, so callers
    # transparently see the resolved (torch-aware) dict on first access.
    global _KERNEL_CONFIGS_CACHE
    if name == "KERNEL_CONFIGS":
        if _KERNEL_CONFIGS_CACHE is None:
            _KERNEL_CONFIGS_CACHE = _build_all_configs()
        return _KERNEL_CONFIGS_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
