"""Shared helpers for kernel config modules.

Torch is imported lazily so that ``import kernel_configs`` works in
environments without a GPU stack (e.g. CI lint runners).
"""

from __future__ import annotations

from typing import Any

DTYPE_NAMES: frozenset[str] = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "bool",
    }
)


_DTYPE_MAP_CACHE: dict[str, Any] | None = None


def _build_dtype_map() -> dict[str, Any]:
    import torch

    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }


def __getattr__(name: str) -> Any:
    # PEP 562: defer the torch import until DTYPE_MAP is actually accessed.
    global _DTYPE_MAP_CACHE
    if name == "DTYPE_MAP":
        if _DTYPE_MAP_CACHE is None:
            _DTYPE_MAP_CACHE = _build_dtype_map()
        return _DTYPE_MAP_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def dtype_bytes(dtype: Any) -> int:
    import torch

    return torch.tensor([], dtype=dtype).element_size()
