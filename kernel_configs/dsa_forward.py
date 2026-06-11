"""Dynamic sparse attention forward kernel config."""

from __future__ import annotations

import math

import torch

import references

from ._utils import dtype_bytes


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    batch = size["batch"]
    seq_len_q = size["seq_len_q"]
    seq_len_kv = size["seq_len_kv"]
    n_heads = size["n_heads"]
    n_heads_kv = size["n_heads_kv"]
    head_dim = size["head_dim"]
    block_size = size["block_size"]

    total_q = batch * seq_len_q
    total_kv = batch * seq_len_kv
    n_heads_block = n_heads_kv

    q = torch.randn(total_q, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv, n_heads_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv, n_heads_kv, head_dim, dtype=dtype, device=device)

    cu_seqlens_q = torch.tensor(
        [i * seq_len_q for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [i * seq_len_kv for i in range(batch + 1)], dtype=torch.int32, device=device
    )
    token2batch_q = torch.repeat_interleave(
        torch.diff(cu_seqlens_q), output_size=total_q
    )

    num_kv_blocks = seq_len_kv // block_size
    block_indices = torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
    block_indices = block_indices.unsqueeze(0).unsqueeze(0).expand(
        total_q, n_heads_block, -1
    ).contiguous()

    scale = 1.0 / math.sqrt(head_dim)

    return {
        "q": q, "k": k, "v": v,
        "block_indices": block_indices,
        "indices_blk_siz": block_size,
        "scale": scale,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "token2batch_q": token2batch_q,
    }


def reference_fn(inputs: dict):
    return references.dsa_forward_ref(**inputs)


def flops_fn(size: dict) -> int:
    # The oracle is CAUSAL (masks ki > qi + offset), so a correct kernel does ~half the QK^T and
    # softmax*V FLOPs of dense attention. The 0.5 triangular factor keeps the roofline floor a true
    # LOWER bound (sl_q <= sl_kv => the real causal fraction is >= 0.5), so an honest causal kernel
    # near peak is not false-rejected.
    dense = size["batch"] * size["seq_len_q"] * size["n_heads"] * (
        2 * size["seq_len_kv"] * size["head_dim"]
        + 2 * size["seq_len_kv"] * size["head_dim"]
    )
    return dense // 2


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (
        size["batch"] * size["seq_len_q"] * size["n_heads"] * size["head_dim"] * eb
        + size["batch"] * size["seq_len_kv"] * size["n_heads_kv"] * size["head_dim"] * eb * 2
        + size["batch"] * size["seq_len_q"] * size["n_heads"] * size["head_dim"] * eb
        + size["batch"] * size["seq_len_q"] * size["n_heads"] * 4
    )
