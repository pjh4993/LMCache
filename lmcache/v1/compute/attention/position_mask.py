# SPDX-License-Identifier: Apache-2.0
"""Position-correct causal attention for CacheBlend's recomputed tokens.

After the check layer, CacheBlend recomputes only a gathered subset of the
tokens (``imp_indices``) and lets them attend to the full, partly cached key
sequence. Calling a standard causal kernel with ``len(q) < len(k)`` aligns the
causal mask to the bottom-right corner: the r-th selected row is treated as if
it sat at position ``N - M + r``, so every selected token that is not in the
trailing run also attends to later (future) tokens. The helpers here mask each
row at its own position instead: row ``i`` sees keys ``0 .. q_positions[i]``.

Two implementations, same semantics:

* FlashAttention-4 (CuTe DSL, bundled with recent vLLM) with a ``mask_mod``
  that reads the per-row limit from an aux tensor. Same kernel family as the
  default path, so the cost of the exact mask is negligible.
* A PyTorch SDPA fallback with an explicit boolean mask, used when FA4 is not
  available (older vLLM, pre-Hopper GPUs, or features FA4 is not asked to
  handle here such as ALiBi, sliding windows or logit soft-capping).

See https://github.com/LMCache/LMCache/issues/1152 and
https://github.com/YaoJiayi/CacheBlend/issues/21 for the discussion.
"""

# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger

logger = init_logger(__name__)

_FA4 = None
_ROW_LIMIT_MASK = None
_FA4_TRIED = False


def _load_fa4():
    """Import FA4 and build the mask_mod once.

    Returns (fa4_func, mask_mod), or (None, None) when unavailable.
    """
    global _FA4, _ROW_LIMIT_MASK, _FA4_TRIED
    if _FA4_TRIED:
        return _FA4, _ROW_LIMIT_MASK
    _FA4_TRIED = True
    try:
        # Third Party
        from vllm.vllm_flash_attn.cute.interface import (
            flash_attn_varlen_func as fa4,  # type: ignore
        )
        import cutlass  # type: ignore
        import cutlass.cute as cute  # type: ignore
        import vllm.vllm_flash_attn.cute.utils as fa4_utils  # type: ignore

        @cute.jit
        def row_limit_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
            # aux_tensors[0]: [batch, max_q] int32, the last key position row q may see
            limit = aux_tensors[0]
            b = fa4_utils.ssa_to_scalar(batch)
            qi = fa4_utils.ssa_to_scalar(q_idx)
            kv = fa4_utils.ssa_to_scalar(kv_idx)
            res = cute.make_rmem_tensor(1, dtype=cutlass.Boolean)
            res[0] = kv <= limit[b, qi]
            return res.load()

        _FA4, _ROW_LIMIT_MASK = fa4, row_limit_mask
    except Exception as e:  # pragma: no cover - depends on the environment
        logger.info(
            "FA4 mask_mod unavailable (%s); CacheBlend will use the SDPA "
            "fallback for position-correct masking",
            e,
        )
    return _FA4, _ROW_LIMIT_MASK


def fa4_mask_available(device: Optional[torch.device] = None) -> bool:
    if torch_device_type != "cuda" or not torch_dev.is_available():
        return False
    major, _ = torch_dev.get_device_capability(device)
    if major < 9:
        return False
    fa4, mask_mod = _load_fa4()
    return fa4 is not None and mask_mod is not None


def position_masked_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference / fallback. q: [M, Hq, D], k/v: [N, Hkv, D], q_positions: [M]."""
    hq, hkv = q.shape[1], k.shape[1]
    kk = k.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)  # [Hq, N, D]
    vv = v.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    n = k.shape[0]
    mask = (
        torch.arange(n, device=q.device)[None, :]
        <= q_positions.to(q.device).long()[:, None]
    )  # [M, N]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        kk[None],
        vv[None],
        attn_mask=mask[None, None],
        scale=softmax_scale,
    )
    return out[0].transpose(0, 1).contiguous()


def position_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    use_fa4: bool = True,
) -> torch.Tensor:
    """Single-sequence attention where row i sees keys 0..q_positions[i].

    q: [M, Hq, D] (rows in any order), k/v: [N, Hkv, D] full key sequence,
    q_positions: [M] integer positions of the rows inside that sequence.
    """
    if use_fa4 and fa4_mask_available(q.device):
        fa4, mask_mod = _load_fa4()
        m, n = q.shape[0], k.shape[0]
        cu_q = torch.tensor([0, m], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, n], dtype=torch.int32, device=q.device)
        limit = q_positions.to(device=q.device, dtype=torch.int32).view(1, m)
        res = fa4(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=m,
            max_seqlen_k=n,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=mask_mod,
            aux_tensors=[limit],
        )
        o = res[0] if isinstance(res, tuple) else res
    else:
        o = position_masked_attention_torch(q, k, v, q_positions, softmax_scale)
    if out is not None:
        out[: o.shape[0]].copy_(o.to(out.dtype))
        return out
    return o
