"""
*Experimental* implementation of FlashAttention in Triton.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math

import torch

from einops import rearrange, repeat

import triton
import triton.language as tl


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Bias, Out, Lse,
    softmax_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh,
    stride_ob, stride_oh,
    nheads, seqlen_k, headdim,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_d
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    # initialize pointer to m and l
    lse_i = -float("inf")
    m_i = -float("inf")
    # load q: it will stay in SRAM throughout
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_d < headdim, other=0.0)
    # -- compute qk ----
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    else:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0)
    qk = tl.sum(q.to(tl.float32)[None, :] * k.to(tl.float32), 1)
    # Need to mask out otherwise the softmax is wrong
    qk += tl.where(offs_n < seqlen_k, 0, float("-inf"))
    if BIAS_TYPE == 'vector':
        bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
        # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
        # can then fuse the mult and add into an fma instruction. But if we have bias we need to
        # to multiply with softmax_scale here.
        qk = qk * softmax_scale + bias
        m = tl.maximum(tl.max(qk, 0), lse_i)
        p = tl.exp(qk - m)
    else:
        m = tl.maximum(tl.max(qk, 0) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m)
    l = tl.sum(p, 0)
    p /= l

    if EVEN_HEADDIM:
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    else:
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0)
    # p = p.to(v.dtype)
    o = tl.sum(p[:, None] * v.to(tl.float32), 0)
    # write back lse
    lse_ptrs = Lse + off_hb + n_block_idx
    tl.store(lse_ptrs, l)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + n_block_idx * headdim + offs_d
    if EVEN_HEADDIM:
        tl.store(out_ptrs, o)
    else:
        tl.store(out_ptrs, o, mask=offs_d < headdim)


def flash_attn_single_query(q, k, v, bias=None, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    assert seqlen_q == 1
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        assert bias.shape[2:] == (1, seqlen_k)
        bias_type = 'vector'
        if bias.shape[:2] == (1, nheads):
            bias = repeat(bias, '1 h ... -> b h ...', b=batch)
        elif bias.shape[:2] == (batch, 1):
            bias = repeat(bias, 'b 1 ... -> b h ...', h=nheads)
        assert bias.shape[:2] == (batch, nheads), 'First 2 dimensions of bias must be broadcastible to (batch, nheads)'
    bias_strides = (bias.stride(0), bias.stride(1)) if has_bias else (0, 0)

    BLOCK = 128
    num_n_blocks = (seqlen_k + BLOCK - 1) // BLOCK
    lse = torch.empty((batch, nheads, num_n_blocks), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q, k, v, bias, o, lse,
        softmax_scale,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *bias_strides,
        o.stride(0), o.stride(2),
        nheads, seqlen_k, d,
        seqlen_q // 32,  seqlen_k // 32, # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # BLOCK_HEADDIM=d,
        bias_type, BLOCK_HEADDIM,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated
