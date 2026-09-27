# SPDX-License-Identifier: Apache-2.0
"""CacheBlend's recomputed rows must attend with a position-correct causal mask.

The exact answer for a gathered subset of rows is the corresponding rows of full
causal attention over the whole sequence (the other rows' queries do not matter).
A causal kernel called with len(q) < len(k) aligns the mask to the bottom-right
corner instead, which lets gathered rows see later tokens.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
from lmcache.v1.compute.attention.position_mask import (
    fa4_mask_available,
    position_masked_attention,
    position_masked_attention_torch,
)


def _full_causal(q, k, v, scale):
    hq, hkv = q.shape[1], k.shape[1]
    kk = k.transpose(0, 1).repeat_interleave(hq // hkv, 0)[None]
    vv = v.transpose(0, 1).repeat_interleave(hq // hkv, 0)[None]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], kk, vv, is_causal=True, scale=scale
    )
    return out[0].transpose(0, 1)


def _bottom_right(q_sel, k, v, scale):
    m, n = q_sel.shape[0], k.shape[0]
    hq, hkv = q_sel.shape[1], k.shape[1]
    kk = k.transpose(0, 1).repeat_interleave(hq // hkv, 0)[None]
    vv = v.transpose(0, 1).repeat_interleave(hq // hkv, 0)[None]
    mask = torch.arange(n)[None, :] <= (n - m + torch.arange(m))[:, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        q_sel.transpose(0, 1)[None], kk, vv, attn_mask=mask[None, None], scale=scale
    )
    return out[0].transpose(0, 1)


def _case(
    n=97, hq=8, hkv=2, d=64, ratio=0.3, seed=0, device="cpu", dtype=torch.float32
):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, hq, d, generator=g).to(device, dtype)
    k = torch.randn(n, hkv, d, generator=g).to(device, dtype)
    v = torch.randn(n, hkv, d, generator=g).to(device, dtype)
    m = max(2, int(n * ratio))
    idx = torch.sort(torch.randperm(n - 1, generator=g)[: m - 1]).values
    idx = torch.cat([idx, torch.tensor([n - 1])]).to(device)  # last token always kept
    return q, k, v, idx


def test_metadata_records_positions():
    md = LMCFlashAttnMetadata(
        query_start_loc=torch.tensor([0, 10], dtype=torch.int32),
        seq_lens=torch.tensor([10]),
        cu_seqlens_k=torch.tensor([0, 10], dtype=torch.int32),
        max_query_len=10,
        max_seq_len=10,
    )
    assert md.q_positions is None
    md.update_from_top_indices(torch.tensor([1, 4, 9]))
    assert md.q_positions.tolist() == [1, 4, 9]
    assert md.max_query_len == 3


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.7])
def test_torch_path_matches_full_causal_rows(ratio):
    q, k, v, idx = _case(ratio=ratio)
    scale = q.shape[-1] ** -0.5
    ref = _full_causal(q, k, v, scale)[idx]
    got = position_masked_attention_torch(q[idx], k, v, idx, scale)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_bottom_right_mask_is_not_exact():
    q, k, v, idx = _case(ratio=0.3)
    scale = q.shape[-1] ** -0.5
    ref = _full_causal(q, k, v, scale)[idx]
    br = _bottom_right(q[idx], k, v, scale)
    # the trailing row is exact, the gathered interior rows are not
    torch.testing.assert_close(br[-1], ref[-1], atol=1e-5, rtol=1e-5)
    assert (br[:-1] - ref[:-1]).abs().max() > 1e-2


@pytest.mark.cuda
@pytest.mark.skipif(
    not fa4_mask_available(), reason="FA4 mask_mod needs vLLM's CuTe FA4 on sm90+"
)
def test_fa4_path_matches_reference():
    q, k, v, idx = _case(
        n=1537,
        hq=28,
        hkv=4,
        d=128,
        ratio=0.3,
        device=torch_device_type,
        dtype=torch.bfloat16,
    )
    scale = q.shape[-1] ** -0.5
    ref = position_masked_attention_torch(
        q[idx].float(), k.float(), v.float(), idx, scale
    )
    got = position_masked_attention(q[idx], k, v, idx, scale)
    assert (got.float() - ref).abs().max() < 2e-2
