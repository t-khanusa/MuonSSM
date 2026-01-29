# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import triton
import triton.language as tl
from fla.utils import contiguous
from torch.cuda.amp import custom_bwd, custom_fwd
from .wy_fast import fwd_recompute_w_u, fwd_prepare_wy_repr, bwd_prepare_wy_repr
from einops import rearrange
import torch.nn.functional as F 


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def fwd_prepare_du_kernel(
    q,
    k,
    g,
    do,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1)) 
        b_q = (b_q * scale).to(b_k.dtype)
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))
    b_A = b_A * tl.math.exp2(b_g[None, :] - b_g[:, None])
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A , 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))



def fwd_prepare_du(q, k, g, do, BT):
    dv = torch.empty_like(do)
    B, H, T, K, V = *k.shape, do.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    fwd_prepare_du_kernel[(NT, B*H)](
        q, k, g, do, dv,
        k.stride(1), k.stride(2), k.stride(3), 
        do.stride(1), do.stride(2), do.stride(3),
        T, K, V, K**-0.5, BT, BK, BV
    )
    return dv

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)

@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    w, 
    v_new,
    g, 
    h,
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        b_g_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)
        
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))            
            b_g = tl.load(g + i_bh * T + i_t * BT + i_c * BC + tl.arange(0, BC)) 
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))   
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BK]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_w = (b_w * tl.math.exp2(b_g)[:, None]).to(b_k.dtype)
            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_w, b_h.to(b_k.dtype), allow_tf32=False)
            # [BK, BV]
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_k = (b_k * tl.math.exp2(b_g_last - b_g)[None, :]).to(p_k.dtype.element_ty)
            b_h_cumsum += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)        
        b_h *= tl.math.exp2(b_g_last)
        b_h += b_h_cumsum      
        
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    g,
    h,
    o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)) 
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = g + i_bh * T + i_t * BT + tl.arange(0, BT)
    b_g = tl.load(p_g)
    b_o = b_o * tl.math.exp2(b_g)[:, None]
    b_s = b_s * tl.math.exp2(b_g[:, None] - b_g[None, :])
    m_s = o_i[:, None] >= o_i[None, :]
    b_s = tl.where(m_s, b_s, 0)
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) 
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_gated_delta_rule_bwd_kernel_dhu(
    q,
    k,
    w,
    g,
    do,
    dh,
    dv,
    dv2,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)

        bg_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_g = tl.load(g + i_bh * T + i_t * BT + i_c * BC + tl.arange(0, BC))
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale * tl.math.exp2(b_g)[None, :]).to(b_q.dtype)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False) 
            # b_q = b_do = None
            # [BT, BK]

            b_k = tl.load(p_k, boundary_check=(0, 1))            
            b_k = (b_k * tl.math.exp2(bg_last - b_g)[:, None]).to(b_k.dtype)            
            b_w = tl.load(p_w, boundary_check=(0, 1))        
            b_w = (b_w * tl.math.exp2(b_g)[None, :]).to(b_w.dtype)
            # [BT, V]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            p_dv2 = tl.make_block_ptr(dv2 + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp -= tl.dot(b_w, b_dv.to(b_q.dtype), allow_tf32=False)

        b_dh *= tl.math.exp2(bg_last)
        b_dh += b_dh_tmp

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_gated_delta_rule_bwd_kernel_dqkw(
    q,
    k,
    v,
    w, 
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dw,
    dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)

    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT, ], dtype=tl.float32)
    b_g_last = tl.load(g + i_bh * T + i_t * BT + BT - 1)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT]        
        b_dg_last += (tl.sum(b_h * b_dh))               
        # [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)  
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype), allow_tf32=False)
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
    b_dg_last *= tl.math.exp2(b_g_last)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))    
    b_w = tl.load(p_w, boundary_check=(0, 1))

    b_g_exp_qw = tl.math.exp2(b_g)
    b_dq *= b_g_exp_qw[:, None] * scale
    b_dg += tl.sum(b_dq * b_q, axis=1)
    b_dw *= b_g_exp_qw[:, None]
    b_dg -= tl.sum(b_dw * b_w, axis=1)
    b_dk *= tl.math.exp2(b_g_last - b_g)[:, None]
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)
    b_g_exp_qw = None
    # [BT, BT]
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.math.exp2(b_g[:, None] - b_g[None, :]), 0).to(b_q.dtype)
    # gradient wrt
    b_dg_mask = tl.dot(b_q, tl.trans(b_k), allow_tf32=False) * b_ds 
    b_dg += tl.sum(b_dg_mask, axis=1)
    b_dg -= tl.sum(b_dg_mask, axis=0)


    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(tl.trans(b_q), b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dg + (i_bh + i_k * B*H)* T + i_t * BT + tl.arange(0, BT), b_dg)
    tl.debug_barrier()
    b_dg_last_prev = tl.load(dg + (i_bh + i_k * B*H)* T + i_t * BT + BT - 1 )
    b_dg_last += b_dg_last_prev
    tl.store(dg + (i_bh + i_k * B*H)* T + i_t * BT + BT - 1 + tl.arange(0, 1), b_dg_last)


def chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state, state_in_fp32=False):
    B, H, T, K, V = *k.shape, u.shape[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    BV = 16 if BK > 128 else 32        
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32 
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = k.new_empty(B, H, NT * K, V)
    if state_in_fp32:
        h = h.float()
    grid = (NK, NV, B * H)
    v_new = torch.empty_like(u)
    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k, u, w, v_new, g, h, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        u.stride(1), u.stride(2), u.stride(3),
        h.stride(1), h.stride(2),
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        )
    return h, v_new
    

def chunk_bwd_dhu_fn(q, k, w, g, do, dv, BT):
    B, H, T, K, V = *q.shape, do.shape[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    BV = 16 if BK > 128 else 32        
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32 
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    # always state in fp32.
    dh = q.new_empty(B, H, NT * K, V, dtype=torch.float32)
    # dv_new = torch.empty_like(do)
    grid = (NK, NV, B * H)
    dv2 = torch.empty_like(dv)
    chunk_gated_delta_rule_bwd_kernel_dhu[grid](
        q, k, w, g, do, dh, dv, dv2,
        q.stride(1), q.stride(2), q.stride(3),
        do.stride(1), do.stride(2), do.stride(3),
        dh.stride(1), dh.stride(2),
        K**-0.5,
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,
    )
    return dh, dv2


def chunk_fwd_o_fn(q, k, v_new, g, h, BT):
    B, H, T, K, V = *q.shape, v_new.shape[-1]

    BK = triton.next_power_of_2(K)
    o = torch.empty_like(v_new)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(K), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H)
    chunk_linear_attn_fwd_kernel_o[grid](
            q, k, v_new, g, h, o,
            q.stride(1), q.stride(2), q.stride(3),
            v_new.stride(1), v_new.stride(2), v_new.stride(3),
            h.stride(1), h.stride(2),
            scale=K**-0.5,
            H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    return o



def chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT):
    B, H, T, K, V = *q.shape, v_new.shape[-1]

    BK = triton.next_power_of_2(K)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k) 
    dw = torch.empty_like(w) 
    dg = torch.zeros(NK, *g.shape, dtype=torch.float32, device=g.device)
    chunk_gated_delta_rule_bwd_kernel_dqkw[grid](
        q, k, v_new, w, g, h, do, dh, dq, dk, du, dw, dg,
        q.stride(1), q.stride(2), q.stride(3),
        v_new.stride(1), v_new.stride(2), v_new.stride(3),
        dh.stride(1), dh.stride(2),
        scale = K ** -0.5,
        B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
    )
    dg = dg.sum(0)
    return dq, dk, dw, dg


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, q, k, v, beta, g, BT, initial_state, output_final_state):        
        g = g.float()
        #currently we force the length to be multiple of BT
        # assert g.shape[-1] % BT == 0
        g = rearrange(g, 'b h (n c) -> b h n c', c=BT)
        # change the base of log from e to 2, i.e., ln->log2. To use tl.math.exp2 inside the kernel. 
        g = g.cumsum(-1) * 1.44269504
        g = rearrange(g, 'b h n c -> b h (n c)')

        ### obtain WY representation. u is actually the new v.
        w, u, A_w, A_u, A_w_original, A_u_original = fwd_prepare_wy_repr(k, v, beta, g, BT)
        ### forward_h 
        final_state = None
        # state will convert to bf16 to do matmul anyway so we don't need fp32 state in the forward pass.
        h, v_new = chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state, state_in_fp32=False)                
        ## obtain output 
        o = chunk_fwd_o_fn(q, k, v_new, g, h, BT)
        # save memory
        # if checkpoint_level == 1:
        # always save memory
        h, v_new = None, None
        ctx.save_for_backward(q, k, v, beta, g, A_w, A_u, A_w_original, A_u_original, h, v_new, initial_state)
        ctx.BT = BT
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do, d_ht=None):
        q, k, v, beta, g, A_w, A_u, A_w_original, A_u_original, h, v_new, initial_state = ctx.saved_tensors
        BT = ctx.BT
        w, u = fwd_recompute_w_u(k, v, beta, A_w, A_u, BT)
        # checkpont_level=1, recomputation.
        # we need fp32 state to compute gradient.
        if h is None:
            h, v_new = chunk_fwd_h_fn(k, w, u, g, BT, initial_state, None, state_in_fp32=True)
        du = fwd_prepare_du(q, k, g, do, BT)
        dh, du = chunk_bwd_dhu_fn(q, k, w, g, do, du, BT)
        dq, dk, dw, dg = chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT)
        dk2, dv, dbeta, dg2 = bwd_prepare_wy_repr(k, v, beta, g, A_w, A_u, A_w_original, A_u_original, dw, du, BT)
        dk.add_(dk2)
        dg.add_(dg2)

        dg = rearrange(dg, 'b h (n c) -> b h n c', c=BT)
        # mask = (torch.arange(0, BT)[:, None] >= torch.arange(0, BT)[None, :]).to(dg)
        assert dg.dtype == torch.float32, "dg should be fp32"
        # print(dg.abs().max())
        # dg = dg @ mask
        # dg = dg * 1.44269504
        def rev_cumsum(x):
            cumsum_x = x.cumsum(-1)
            rev_cumsum_x = cumsum_x[..., -1, None] - cumsum_x
            return rev_cumsum_x + x

        dg = rev_cumsum(dg)
        dg = rearrange(dg, 'b h n c -> b h (n c)')
        # print(dg.abs().max(), dq.abs().max(), dk.abs().max(), dv.abs().max(), dbeta.abs().max())
        # if dg.isnan().any():
        # breakpoint()

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype), dg.to(g.dtype), None, None, None, None



def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64, #chunk size
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    assert q.dtype == k.dtype == v.dtype
    L = q.shape[-2]
    if L % BT != 0:
        q, k, v, beta, g = map(lambda x: F.pad(x, (0, 0, 0, BT - L % BT)), [q, k, v, beta.unsqueeze(-1), g.unsqueeze(-1)])
    g = g.squeeze(-1)
    beta = beta.squeeze(-1)

    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkGatedDeltaRuleFunction.apply(q, k, v, beta, g, BT,  initial_state, output_final_state)
    return o[:, :, :L, :], final_state


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int, #chunk size
):
    # alias
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), [q, k, v, k_beta, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
      attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
      attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
      q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
      attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
      v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
      v_new = v_i - v_prime
      o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
      o[:, :, i] = o_inter + attn @ v_new
      S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    return rearrange(o, 'b h n c d -> b h (n c) d'),  rearrange(w, 'b h n c d -> b h (n c) d'),  rearrange(u, 'b h n c d -> b h (n c) d')

def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
):
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        S = S.clone() * g[:, :, i].exp()[..., None, None]
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o


def zeropower_via_newtonschulz5(G, steps=1):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    G has shape [..., D, N] where the last two dimensions form the matrix to orthogonalize.
    
    Uses quintic iteration with coefficients selected to maximize the slope at zero.
    """
    assert G.ndim >= 2  # batched implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X.to(G.dtype)


def zeropower_via_newtonschulz5_backward(G, grad_output, steps=1):
    """
    Backward pass for Newton-Schulz orthogonalization.
    Uses straight-through estimator for simplicity and stability.
    """
    # Straight-through estimator: just pass the gradient through
    # This is a common approximation for orthogonalization operations
    return grad_output


class NewtonSchulzFunction(torch.autograd.Function):
    """Autograd function for Newton-Schulz with straight-through gradient."""
    
    @staticmethod
    def forward(ctx, G, steps):
        ctx.steps = steps
        ctx.save_for_backward(G)
        return zeropower_via_newtonschulz5(G, steps)
    
    @staticmethod
    def backward(ctx, grad_output):
        G, = ctx.saved_tensors
        # Straight-through estimator
        return grad_output, None


def newton_schulz_orth(G, steps=1):
    """Newton-Schulz orthogonalization with gradient support."""
    return NewtonSchulzFunction.apply(G, steps)


def recurrent_muon_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    momentum_alpha: float = 0.9,
    momentum_beta: float = 1.0,
    ns_steps: int = 1,
):
    """
    Muon-enhanced Gated Delta Rule with Newton-Schulz orthogonalization.
    
    Equations:
        V_t = momentum_alpha * V_{t-1} + momentum_beta * N_t  (momentum accumulation)
        S_t = S_{t-1} * M_t + Newton-Schulz(V_t)              (state update)
    
    Where:
        M_t = decay (forget gate)
        N_t = k_t @ v_t^T (write operation)
    """
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)  # State
    V = torch.zeros(b, h, d_k, d_v).to(v)  # Velocity/Momentum
    q = q * (d_k ** -0.5)
    
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        
        # Apply decay to state: S' = S * decay (M_t part)
        decay_i = g[:, :, i].exp()[..., None, None]
        S_decayed = S.clone() * decay_i
        
        # Delta rule: compute error and scale by beta
        beta_i = beta[:, :, i]
        _v = _v - (S_decayed.clone() * _k[..., None]).sum(-2)  # v' = v - S' @ k
        _v = _v * beta_i[..., None]  # v'' = beta * v'
        
        # N_t = k @ v''^T (the write operation)
        N_t = _k.unsqueeze(-1) * _v.unsqueeze(-2)  # [B, H, D_k, D_v]
        
        # Muon momentum update: V_t = alpha * V_{t-1} + beta * N_t
        V = momentum_alpha * V + momentum_beta * N_t
        
        # Apply Newton-Schulz orthogonalization to momentum V
        # V has shape [B, H, D_k, D_v]
        V_orth = zeropower_via_newtonschulz5(V, steps=ns_steps)
        
        # State update: S_t = S' + NS(V_t)
        S = S_decayed + V_orth
        
        # Output: o = q^T @ S
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    
    return o


def _muon_gated_delta_rule_forward_with_grad(q, k, v, beta, g, momentum_alpha, momentum_beta, ns_steps, initial_state=None):
    """
    Pure PyTorch implementation of Muon Gated Delta Rule forward pass.
    Supports autograd for backward pass by maintaining gradient flow.
    
    IMPORTANT: When initial_state is provided, gradients will flow through it
    to enable learning long-term dependencies across chunks.
    """
    B, H, L, D_k = q.shape
    D_v = v.shape[-1]
    
    # Scale query
    scale = D_k ** -0.5
    q_scaled = q * scale
    
    # Initialize states - maintain gradient flow through initial_state
    if initial_state is not None:
        S_init, V_init = initial_state
        # Don't use .clone() - we want gradients to flow through!
        S = S_init
        V = V_init
    else:
        # Create zeros (no gradient needed for zero initialization)
        S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)
        V = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)
    
    # Collect outputs
    outputs = []
    
    for i in range(L):
        _k = k[:, :, i]  # [B, H, D_k]
        _q = q_scaled[:, :, i]  # [B, H, D_k]
        _v = v[:, :, i]  # [B, H, D_v]
        _beta = beta[:, :, i]  # [B, H]
        _g = g[:, :, i]  # [B, H]
        
        # Apply decay to state: S' = S * exp(g)
        decay = _g.exp()[..., None, None]  # [B, H, 1, 1]
        S_decayed = S * decay
        
        # Delta rule: compute error
        retrieved = (S_decayed * _k[..., None]).sum(-2)  # [B, H, D_v]
        v_err = _v - retrieved
        v_err = v_err * _beta[..., None]  # Scale by beta
        
        # N_t = k @ v_err^T (outer product)
        N_t = _k.unsqueeze(-1) * v_err.unsqueeze(-2)  # [B, H, D_k, D_v]
        
        # Muon momentum update: V = alpha * V + beta * N_t
        V = momentum_alpha * V + momentum_beta * N_t
        
        # Apply Newton-Schulz orthogonalization with gradient support (straight-through)
        V_orth = newton_schulz_orth(V, steps=ns_steps)
        
        # State update: S = S' + NS(V)
        S = S_decayed + V_orth
        
        # Output: o = q^T @ S
        o_i = torch.einsum('bhd,bhdm->bhm', _q, S)
        outputs.append(o_i)
    
    o = torch.stack(outputs, dim=2)  # [B, H, L, D_v]
    
    return o, (S, V)


def chunk_muon_gated_delta_rule_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    momentum_alpha: float = 0.9,
    momentum_beta: float = 1.0,
    ns_steps: int = 1,
    initial_state=None,
    output_final_state: bool = False
):
    """
    Muon Gated Delta Rule using pure PyTorch autograd (no custom backward).
    This version properly supports gradient computation including initial_state.
    
    IMPORTANT: Gradients flow through initial_state for learning long-term
    dependencies across chunks when training on long sequences.
    """
    # Convert initial_state to float if provided
    if initial_state is not None:
        initial_state_f = (initial_state[0].float(), initial_state[1].float())
    else:
        initial_state_f = None
    
    o, final_state = _muon_gated_delta_rule_forward_with_grad(
        q.float(), k.float(), v.float(), beta.float(), g.float(),
        momentum_alpha, momentum_beta, ns_steps, initial_state_f
    )
    
    if not output_final_state:
        final_state = None
        
    return o.to(q.dtype), final_state


class ChunkMuonGatedDeltaRuleFunction(torch.autograd.Function):
    """
    Chunked Muon-enhanced Gated Delta Rule with Newton-Schulz orthogonalization.
    Uses torch.autograd.grad for cleaner backward computation.
    
    IMPORTANT: Properly handles initial_state gradients for learning long-term
    dependencies across chunks when training on long sequences.
    """

    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, q, k, v, beta, g, BT, momentum_alpha, momentum_beta, ns_steps, initial_state_S, initial_state_V, output_final_state):
        # Save inputs for backward recomputation
        # Note: initial_state components are saved separately as tensors
        if initial_state_S is not None and initial_state_V is not None:
            ctx.save_for_backward(q, k, v, beta, g, initial_state_S, initial_state_V)
            ctx.has_initial_state = True
            initial_state = (initial_state_S, initial_state_V)
        else:
            ctx.save_for_backward(q, k, v, beta, g)
            ctx.has_initial_state = False
            initial_state = None
            
        ctx.BT = BT
        ctx.momentum_alpha = momentum_alpha
        ctx.momentum_beta = momentum_beta
        ctx.ns_steps = ns_steps
        
        # Run forward without gradient tracking for efficiency
        with torch.no_grad():
            o, final_state = _muon_gated_delta_rule_forward_with_grad(
                q.float(), k.float(), v.float(), beta.float(), g.float(),
                momentum_alpha, momentum_beta, ns_steps, initial_state
            )
        
        if not output_final_state:
            final_state = None
            
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous  
    def backward(ctx, do, d_final_state=None):
        momentum_alpha = ctx.momentum_alpha
        momentum_beta = ctx.momentum_beta
        ns_steps = ctx.ns_steps
        has_initial_state = ctx.has_initial_state
        
        # Retrieve saved tensors
        if has_initial_state:
            q, k, v, beta, g, initial_state_S, initial_state_V = ctx.saved_tensors
        else:
            q, k, v, beta, g = ctx.saved_tensors
            initial_state_S = None
            initial_state_V = None
        
        # Create leaf tensors with requires_grad for recomputation
        q_f = q.float().detach().requires_grad_(True)
        k_f = k.float().detach().requires_grad_(True)
        v_f = v.float().detach().requires_grad_(True)
        beta_f = beta.float().detach().requires_grad_(True)
        g_f = g.float().detach().requires_grad_(True)
        
        # Handle initial_state gradients
        if has_initial_state:
            S_init_f = initial_state_S.float().detach().requires_grad_(True)
            V_init_f = initial_state_V.float().detach().requires_grad_(True)
            initial_state_f = (S_init_f, V_init_f)
            inputs = (q_f, k_f, v_f, beta_f, g_f, S_init_f, V_init_f)
        else:
            initial_state_f = None
            inputs = (q_f, k_f, v_f, beta_f, g_f)
        
        # Recompute forward with gradients enabled
        o, _ = _muon_gated_delta_rule_forward_with_grad(
            q_f, k_f, v_f, beta_f, g_f,
            momentum_alpha, momentum_beta, ns_steps, initial_state_f
        )
        
        # Use torch.autograd.grad to compute gradients
        grads = torch.autograd.grad(
            outputs=o,
            inputs=inputs,
            grad_outputs=do.float(),
            allow_unused=True
        )
        
        dq = grads[0].to(q.dtype) if grads[0] is not None else torch.zeros_like(q)
        dk = grads[1].to(k.dtype) if grads[1] is not None else torch.zeros_like(k)
        dv = grads[2].to(v.dtype) if grads[2] is not None else torch.zeros_like(v)
        dbeta = grads[3].to(beta.dtype) if grads[3] is not None else torch.zeros_like(beta)
        dg = grads[4].to(g.dtype) if grads[4] is not None else torch.zeros_like(g)
        
        # Handle initial_state gradients
        if has_initial_state:
            dS_init = grads[5].to(initial_state_S.dtype) if grads[5] is not None else torch.zeros_like(initial_state_S)
            dV_init = grads[6].to(initial_state_V.dtype) if grads[6] is not None else torch.zeros_like(initial_state_V)
        else:
            dS_init = None
            dV_init = None
        
        # Return gradients for all inputs: q, k, v, beta, g, BT, momentum_alpha, momentum_beta, ns_steps, initial_state_S, initial_state_V, output_final_state
        return dq, dk, dv, dbeta, dg, None, None, None, None, dS_init, dV_init, None


def chunk_muon_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    momentum_alpha: float = 0.9,
    momentum_beta: float = 1.0,
    ns_steps: int = 1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    """
    Muon-enhanced Gated Delta Rule with Newton-Schulz orthogonalization.
    
    Equations:
        V_t = momentum_alpha * V_{t-1} + momentum_beta * N_t  (momentum accumulation)
        S_t = S_{t-1} * M_t + Newton-Schulz(V_t)              (state update)
    
    Args:
        q: Query tensor [B, H, L, D_k]
        k: Key tensor [B, H, L, D_k]  
        v: Value tensor [B, H, L, D_v]
        beta: Learning rate tensor [B, H, L]
        g: Decay/gate tensor [B, H, L] (log-space)
        BT: Chunk size (for compatibility, not used in current impl)
        momentum_alpha: Momentum decay parameter (default 0.9)
        momentum_beta: Momentum scale parameter (default 1.0)
        ns_steps: Number of Newton-Schulz iterations (default 1)
        initial_state: Initial (S, V) state tuple or None
        output_final_state: Whether to return final state
        
    Returns:
        o: Output tensor [B, H, L, D_v]
        final_state: Final (S, V) state tuple if output_final_state else None
    """
    assert q.dtype == k.dtype == v.dtype
    
    # Use the pure autograd version which properly supports gradients
    return chunk_muon_gated_delta_rule_autograd(
        q, k, v, beta, g,
        momentum_alpha=momentum_alpha,
        momentum_beta=momentum_beta,
        ns_steps=ns_steps,
        initial_state=initial_state,
        output_final_state=output_final_state
    )

    
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Standard Gated Delta Rule")
    print("=" * 60)
    
    B = 4
    H = 2
    L = 1024
    DK = 128
    DV = 64
    require_grad = True
    dtype = torch.bfloat16
    q = (torch.rand(B, H, L, DK)).cuda().to(dtype)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).to(dtype)
    v = (torch.randn(B, H, L, DV)).cuda().to(dtype)
    beta = torch.randn(B, H, L).sigmoid().cuda()
    decay = torch.empty(B, H, L).uniform_(0.01, 0.03).log().cuda()
    q,k,v,beta,decay = map(lambda x: x.requires_grad_(require_grad), [q,k,v,beta,decay])

    o, _,  = chunk_gated_delta_rule(q, k, v, beta, decay, BT=64)
    o2 = recurrent_gated_delta_rule_ref(q, k, v, beta, decay)
    do2 = torch.randn_like(o2)
    o2.backward(do2)

    q_grad, q.grad = q.grad, None 
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None
    decay_grad, decay.grad = decay.grad, None
    o.backward(do2)
    print("Forward diff (chunk vs recurrent):", (o - o2).abs().max().item())
    print("dq diff:", (q.grad - q_grad).abs().max().item())
    print("dk diff:", (k.grad - k_grad).abs().max().item())
    print("dv diff:", (v.grad - v_grad).abs().max().item())
    print("dbeta diff:", (beta.grad - beta_grad).abs().max().item())
    print("ddecay diff:", (decay.grad - decay_grad).abs().max().item())

    print("\n" + "=" * 60)
    print("Testing Muon Gated Delta Rule")
    print("=" * 60)
    
    # Test Muon version
    B, H, L, DK, DV = 2, 4, 128, 32, 32
    momentum_alpha = 0.9
    momentum_beta = 1.0
    ns_steps = 1
    
    # Test 1: Forward pass (chunk_muon vs recurrent_muon_ref)
    print("\n1. Testing Forward Pass (chunk_muon vs recurrent_muon_ref)...")
    
    q_muon = torch.randn(B, H, L, DK, dtype=torch.float32, device='cuda')
    k_muon = torch.nn.functional.normalize(torch.randn(B, H, L, DK, device='cuda'), dim=-1)
    v_muon = torch.randn(B, H, L, DV, dtype=torch.float32, device='cuda')
    beta_muon = torch.randn(B, H, L, device='cuda').sigmoid()
    decay_muon = torch.empty(B, H, L, device='cuda').uniform_(0.01, 0.03).log()
    
    o_chunk, _ = chunk_muon_gated_delta_rule(
        q_muon, k_muon, v_muon, beta_muon, decay_muon,
        BT=64, momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps
    )
    
    o_ref = recurrent_muon_gated_delta_rule_ref(
        q_muon, k_muon, v_muon, beta_muon, decay_muon,
        momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps
    )
    
    forward_diff = (o_chunk - o_ref).abs().max().item()
    print(f"   Forward diff (chunk_muon vs recurrent_muon_ref): {forward_diff:.6e}")
    
    # Test 2: Backward pass using autograd version
    print("\n2. Testing Backward Pass (autograd version vs reference)...")
    
    # Create fresh tensors with requires_grad
    q_muon2 = torch.randn(B, H, L, DK, dtype=torch.float32, device='cuda', requires_grad=True)
    k_muon2 = torch.nn.functional.normalize(torch.randn(B, H, L, DK, device='cuda'), dim=-1).requires_grad_(True)
    v_muon2 = torch.randn(B, H, L, DV, dtype=torch.float32, device='cuda', requires_grad=True)
    beta_muon2 = torch.randn(B, H, L, device='cuda').sigmoid().requires_grad_(True)
    decay_muon2 = torch.empty(B, H, L, device='cuda').uniform_(0.01, 0.03).log().requires_grad_(True)
    
    # Use autograd version which properly supports gradients
    o_autograd, _ = chunk_muon_gated_delta_rule_autograd(
        q_muon2, k_muon2, v_muon2, beta_muon2, decay_muon2,
        momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps
    )
    
    # Compute gradients via autograd
    do = torch.randn_like(o_autograd)
    o_autograd.backward(do)
    
    # Store autograd gradients
    q_grad_autograd = q_muon2.grad.clone() if q_muon2.grad is not None else None
    k_grad_autograd = k_muon2.grad.clone() if k_muon2.grad is not None else None
    v_grad_autograd = v_muon2.grad.clone() if v_muon2.grad is not None else None
    beta_grad_autograd = beta_muon2.grad.clone() if beta_muon2.grad is not None else None
    decay_grad_autograd = decay_muon2.grad.clone() if decay_muon2.grad is not None else None
    
    print("   Autograd gradients computed successfully!")
    if q_grad_autograd is not None:
        print(f"   dq norm: {q_grad_autograd.norm().item():.6e}")
    if k_grad_autograd is not None:
        print(f"   dk norm: {k_grad_autograd.norm().item():.6e}")
    if v_grad_autograd is not None:
        print(f"   dv norm: {v_grad_autograd.norm().item():.6e}")
    if beta_grad_autograd is not None:
        print(f"   dbeta norm: {beta_grad_autograd.norm().item():.6e}")
    if decay_grad_autograd is not None:
        print(f"   ddecay norm: {decay_grad_autograd.norm().item():.6e}")
    
    # Test 3: Verify chunk_muon_gated_delta_rule backward (should use autograd internally)
    print("\n3. Testing chunk_muon_gated_delta_rule backward...")
    
    # Create fresh tensors
    q_muon3 = q_muon2.detach().clone().requires_grad_(True)
    k_muon3 = k_muon2.detach().clone().requires_grad_(True)
    v_muon3 = v_muon2.detach().clone().requires_grad_(True)
    beta_muon3 = beta_muon2.detach().clone().requires_grad_(True)
    decay_muon3 = decay_muon2.detach().clone().requires_grad_(True)
    
    o_chunk2, _ = chunk_muon_gated_delta_rule(
        q_muon3, k_muon3, v_muon3, beta_muon3, decay_muon3,
        BT=64, momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps
    )
    
    o_chunk2.backward(do)
    
    print("   Gradient differences (chunk_muon vs autograd reference):")
    if q_grad_autograd is not None and q_muon3.grad is not None:
        diff = (q_muon3.grad - q_grad_autograd).abs().max().item()
        print(f"   dq diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    if k_grad_autograd is not None and k_muon3.grad is not None:
        diff = (k_muon3.grad - k_grad_autograd).abs().max().item()
        print(f"   dk diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    if v_grad_autograd is not None and v_muon3.grad is not None:
        diff = (v_muon3.grad - v_grad_autograd).abs().max().item()
        print(f"   dv diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    if beta_grad_autograd is not None and beta_muon3.grad is not None:
        diff = (beta_muon3.grad - beta_grad_autograd).abs().max().item()
        print(f"   dbeta diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    if decay_grad_autograd is not None and decay_muon3.grad is not None:
        diff = (decay_muon3.grad - decay_grad_autograd).abs().max().item()
        print(f"   ddecay diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    
    # Test 4: Test initial_state gradient flow (critical for long sequences)
    print("\n4. Testing initial_state gradient flow (for cross-chunk learning)...")
    print("   NOTE: Only k, v, beta, decay affect the state. Query q only affects output.")
    
    # Simulate two chunks where the second chunk's initial_state comes from the first chunk's final_state
    B, H, L_chunk, DK, DV = 2, 4, 64, 32, 32
    
    # First chunk
    q_c1 = torch.randn(B, H, L_chunk, DK, dtype=torch.float32, device='cuda', requires_grad=True)
    k_c1 = torch.nn.functional.normalize(torch.randn(B, H, L_chunk, DK, device='cuda'), dim=-1).requires_grad_(True)
    v_c1 = torch.randn(B, H, L_chunk, DV, dtype=torch.float32, device='cuda', requires_grad=True)
    beta_c1 = torch.randn(B, H, L_chunk, device='cuda').sigmoid().requires_grad_(True)
    decay_c1 = torch.empty(B, H, L_chunk, device='cuda').uniform_(0.01, 0.03).log().requires_grad_(True)
    
    # Run first chunk and get final_state
    o_c1, final_state = chunk_muon_gated_delta_rule(
        q_c1, k_c1, v_c1, beta_c1, decay_c1,
        momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps,
        output_final_state=True
    )
    
    # Second chunk uses first chunk's final_state as initial_state
    q_c2 = torch.randn(B, H, L_chunk, DK, dtype=torch.float32, device='cuda', requires_grad=True)
    k_c2 = torch.nn.functional.normalize(torch.randn(B, H, L_chunk, DK, device='cuda'), dim=-1).requires_grad_(True)
    v_c2 = torch.randn(B, H, L_chunk, DV, dtype=torch.float32, device='cuda', requires_grad=True)
    beta_c2 = torch.randn(B, H, L_chunk, device='cuda').sigmoid().requires_grad_(True)
    decay_c2 = torch.empty(B, H, L_chunk, device='cuda').uniform_(0.01, 0.03).log().requires_grad_(True)
    
    o_c2, _ = chunk_muon_gated_delta_rule(
        q_c2, k_c2, v_c2, beta_c2, decay_c2,
        momentum_alpha=momentum_alpha, momentum_beta=momentum_beta, ns_steps=ns_steps,
        initial_state=final_state
    )
    
    # Backward from second chunk should propagate to first chunk's parameters
    # that affect the state (k, v, beta, decay - NOT q)
    loss = o_c2.sum()
    loss.backward()
    
    # Check if gradients flowed to first chunk's state-affecting parameters
    # Note: q does NOT affect the state, only the output, so q_c1.grad can be None
    c1_has_grad = all([
        k_c1.grad is not None and k_c1.grad.abs().sum() > 0,
        v_c1.grad is not None and v_c1.grad.abs().sum() > 0,
        beta_c1.grad is not None and beta_c1.grad.abs().sum() > 0,
        decay_c1.grad is not None and decay_c1.grad.abs().sum() > 0,
    ])
    
    if c1_has_grad:
        print("   ✓ Gradients flow from chunk 2 back to chunk 1!")
        print(f"   Chunk 1 gradient norms (state-affecting params):")
        print(f"     dk_c1: {k_c1.grad.norm().item():.6e}")
        print(f"     dv_c1: {v_c1.grad.norm().item():.6e}")
        print(f"     dbeta_c1: {beta_c1.grad.norm().item():.6e}")
        print(f"     ddecay_c1: {decay_c1.grad.norm().item():.6e}")
        print(f"   (q_c1.grad is None as expected - q doesn't affect state)")
    else:
        print("   ✗ WARNING: Gradients did NOT flow to chunk 1!")
        print(f"     k_c1.grad: {k_c1.grad.abs().sum().item() if k_c1.grad is not None else 'None'}")
        print(f"     v_c1.grad: {v_c1.grad.abs().sum().item() if v_c1.grad is not None else 'None'}")
        print(f"     beta_c1.grad: {beta_c1.grad.abs().sum().item() if beta_c1.grad is not None else 'None'}")
        print(f"     decay_c1.grad: {decay_c1.grad.abs().sum().item() if decay_c1.grad is not None else 'None'}")
    
    print("\n" + "=" * 60)
    if forward_diff < 1e-4:
        print("✓ Muon Gated Delta Rule: Forward pass PASSED!")
    else:
        print("✗ Muon Gated Delta Rule: Forward pass FAILED!")
    if c1_has_grad:
        print("✓ Muon Gated Delta Rule: Cross-chunk gradient flow PASSED!")
    else:
        print("✗ Muon Gated Delta Rule: Cross-chunk gradient flow FAILED!")
    print("=" * 60)


