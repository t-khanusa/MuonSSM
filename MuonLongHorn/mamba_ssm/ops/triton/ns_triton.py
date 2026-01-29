import torch
import triton
import triton.language as tl

# ============================================================================
# NATIVE LAYOUT KERNEL: Works directly on [B, L, D, N] without reshape
# ============================================================================

@triton.jit
def _fused_ns_native_kernel(
    X_ptr, Out_ptr,
    stride_b, stride_l, stride_d, stride_n,  # Strides for [B, L, D, N]
    D: tl.constexpr,      # 3072
    N: tl.constexpr,      # 16
    BLOCK_D: tl.constexpr # 128
):
    """
    Fused Newton-Schulz Step for NATIVE [B, L, D, N] layout.
    
    Input: X[b, l, :, :] is a [D, N] matrix
    We compute NS on the TRANSPOSED view [N, D] without copying.
    
    NS iteration: X_new = a*X + (b*A + c*A²)@X where A = X @ X.T
    For [N, D] matrix: A is [N, N]
    """
    # 2D grid: (batch_idx, seq_idx)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Base offset for this [D, N] matrix
    base_offset = batch_idx * stride_b + seq_idx * stride_l
    X_base = X_ptr + base_offset
    Out_base = Out_ptr + base_offset
    
    # =========================================================
    # PHASE 1: Compute Gram Matrix A = X @ X.T (where X is [N, D])
    # We read [D, N] as [N, D] via strided access
    # =========================================================
    
    acc_A = tl.zeros([N, N], dtype=tl.float32)
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        
        # Load X as [N, BLOCK_D] by reading [D, N] with transposed strides
        # Physical layout is [D, N], we want logical [N, D]
        # ptr[n, d] = base + d * stride_d + n * stride_n
        ptrs_x = X_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        
        # A += X @ X.T where X is [N, BLOCK_D]
        acc_A += tl.dot(x_tile, tl.trans(x_tile))
    
    # =========================================================
    # PHASE 2: Compute B = b*A + c*A²
    # =========================================================
    
    a_val = 3.4445
    b_val = -4.7750
    c_val = 2.0315
    
    A = acc_A
    A_sq = tl.dot(A, A)
    B = b_val * A + c_val * A_sq
    
    # =========================================================
    # PHASE 3: Update X_new = a*X + B@X
    # Store result in same [D, N] layout
    # =========================================================
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        
        # Load X as [N, BLOCK_D]
        ptrs_x = X_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        
        # Compute: X_new = a*X + B@X
        bx_tile = tl.dot(B, x_tile)
        x_new = a_val * x_tile + bx_tile
        
        # Store result in [D, N] layout (same transposed access)
        ptrs_out = Out_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        tl.store(ptrs_out, x_new.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_ns_native_backward_kernel(
    X_ptr, dY_ptr, dX_ptr,
    stride_b, stride_l, stride_d, stride_n,
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """
    Backward kernel for native [B, L, D, N] layout.
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    base_offset = batch_idx * stride_b + seq_idx * stride_l
    X_base = X_ptr + base_offset
    dY_base = dY_ptr + base_offset
    dX_base = dX_ptr + base_offset
    
    # =========================================================
    # PHASE 1: Accumulate A = X @ X.T and dB = dY @ X.T
    # =========================================================
    
    acc_A = tl.zeros([N, N], dtype=tl.float32)
    acc_dB = tl.zeros([N, N], dtype=tl.float32)
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        
        # Load X and dY as [N, BLOCK_D] via transposed access
        ptrs_x = X_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        ptrs_dy = dY_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        dy_tile = tl.load(ptrs_dy, mask=mask, other=0.0).to(tl.float32)
        
        acc_A += tl.dot(x_tile, tl.trans(x_tile))
        acc_dB += tl.dot(dy_tile, tl.trans(x_tile))
    
    # =========================================================
    # PHASE 2: Compute gradients
    # =========================================================
    
    a_val = 3.4445
    b_val = -4.7750
    c_val = 2.0315
    
    A = acc_A
    A_sq = tl.dot(A, A)
    B = b_val * A + c_val * A_sq
    
    dB = acc_dB
    dB_A = tl.dot(dB, A)
    A_dB = tl.dot(A, dB)
    dA = b_val * dB + c_val * (dB_A + A_dB)
    dA_sym = dA + tl.trans(dA)
    
    B_T = tl.trans(B)
    
    # =========================================================
    # PHASE 3: Compute dX = a*dY + B.T @ dY + dA_sym @ X
    # =========================================================
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        
        ptrs_x = X_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        ptrs_dy = dY_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        ptrs_dx = dX_base + (offs_d[None, :] * stride_d) + (offs_n[:, None] * stride_n)
        
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        dy_tile = tl.load(ptrs_dy, mask=mask, other=0.0).to(tl.float32)
        
        term1 = tl.dot(B_T, dy_tile)
        term2 = tl.dot(dA_sym, x_tile)
        dx_tile = a_val * dy_tile + term1 + term2
        
        tl.store(ptrs_dx, dx_tile.to(tl.bfloat16), mask=mask)


def run_fused_ns_native(X, steps=1, return_intermediates=False):
    """
    Fused NS kernel for NATIVE [B, L, D, N] layout - NO RESHAPE NEEDED!
    
    Args:
        X: Input tensor [B, L, D, N] where D > N (tall matrices)
        steps: Number of NS iterations
        return_intermediates: If True, return (output, X_normed, norm) for backward
    
    Returns:
        Output tensor [B, L, D, N] (normalized, no denormalization)
    """
    assert X.dim() == 4, f"Expected 4D tensor [B, L, D, N], got {X.dim()}D"
    B, L, D, N = X.shape
    assert N == 16, f"Kernel optimized for N=16, got N={N}"
    
    # Per-matrix normalization (matching PyTorch impl)
    # X shape: [B, L, D, N], norm over (D, N) dims
    eps = 1e-7
    norm = X.float().norm(dim=(2, 3), keepdim=True) + eps  # [B, L, 1, 1]
    X_normed = (X.float() / norm).to(X.dtype)
    
    Out = torch.empty_like(X_normed)
    
    BLOCK_D = 128 if D >= 128 else 64
    
    # 2D grid: (B, L)
    grid = (B, L)
    
    # Get strides
    stride_b, stride_l, stride_d, stride_n = X_normed.stride()
    
    # Run NS iteration(s)
    curr_X = X_normed
    for _ in range(steps):
        _fused_ns_native_kernel[grid](
            curr_X, Out,
            stride_b, stride_l, stride_d, stride_n,
            D=D, N=N, BLOCK_D=BLOCK_D
        )
        curr_X = Out
    
    if return_intermediates:
        return Out, X_normed, norm
    return Out


def run_fused_ns_native_backward(X_normed, dY, norm, steps=1):
    """
    Backward for native layout fused NS.
    
    Args:
        X_normed: Normalized input [B, L, D, N]
        dY: Upstream gradient [B, L, D, N]
        norm: Normalization factors [B, L, 1, 1]
        steps: Number of NS iterations
    
    Returns:
        dX: Gradient w.r.t. original input [B, L, D, N]
    """
    B, L, D, N = X_normed.shape
    assert N == 16, f"Kernel optimized for N=16, got N={N}"
    
    dX_normed = torch.empty_like(X_normed)
    
    BLOCK_D = 128
    grid = (B, L)
    
    stride_b, stride_l, stride_d, stride_n = X_normed.stride()
    
    _fused_ns_native_backward_kernel[grid](
        X_normed, dY, dX_normed,
        stride_b, stride_l, stride_d, stride_n,
        D=D, N=N, BLOCK_D=BLOCK_D
    )
    
    # Chain rule through normalization
    dX = (dX_normed.float() / norm).to(X_normed.dtype)
    
    return dX


# ============================================================================
# LEGACY: Original [B*L, N, D] layout kernels (kept for compatibility)
# ============================================================================

@triton.jit
def _fused_ns_step_kernel(
    X_ptr, Out_ptr,
    stride_batch, stride_n, stride_d,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """Original kernel for [B*L, N, D] layout."""
    pid = tl.program_id(0)
    
    off_batch = pid * stride_batch
    X_base = X_ptr + off_batch
    Out_base = Out_ptr + off_batch

    acc_A = tl.zeros([N, N], dtype=tl.float32)
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        ptrs_x = X_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        acc_A += tl.dot(x_tile, tl.trans(x_tile))

    a_val = 3.4445
    b_val = -4.7750
    c_val = 2.0315
    
    A = acc_A
    A_sq = tl.dot(A, A)
    B = b_val * A + c_val * A_sq
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        ptrs_x = X_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        bx_tile = tl.dot(B, x_tile)
        x_new = a_val * x_tile + bx_tile
        ptrs_out = Out_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        tl.store(ptrs_out, x_new.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_ns_backward_kernel(
    X_ptr, dY_ptr, dX_ptr,
    stride_batch, stride_n, stride_d,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """Original backward kernel for [B*L, N, D] layout."""
    pid = tl.program_id(0)
    
    off_batch = pid * stride_batch
    X_base = X_ptr + off_batch
    dY_base = dY_ptr + off_batch
    dX_base = dX_ptr + off_batch

    acc_A = tl.zeros([N, N], dtype=tl.float32)
    acc_dB = tl.zeros([N, N], dtype=tl.float32)
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        ptrs_x = X_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        ptrs_dy = dY_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        dy_tile = tl.load(ptrs_dy, mask=mask, other=0.0).to(tl.float32)
        acc_A += tl.dot(x_tile, tl.trans(x_tile))
        acc_dB += tl.dot(dy_tile, tl.trans(x_tile))

    a_val = 3.4445
    b_val = -4.7750
    c_val = 2.0315
    
    A = acc_A
    A_sq = tl.dot(A, A)
    B = b_val * A + c_val * A_sq
    
    dB = acc_dB
    dB_A = tl.dot(dB, A)
    A_dB = tl.dot(A, dB)
    dA = b_val * dB + c_val * (dB_A + A_dB)
    dA_sym = dA + tl.trans(dA)
    B_T = tl.trans(B)
    
    for k in range(0, D, BLOCK_D):
        offs_d = k + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, N)
        mask = offs_d[None, :] < D
        ptrs_x = X_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        ptrs_dy = dY_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        ptrs_dx = dX_base + (offs_n[:, None] * stride_n) + (offs_d[None, :] * stride_d)
        x_tile = tl.load(ptrs_x, mask=mask, other=0.0).to(tl.float32)
        dy_tile = tl.load(ptrs_dy, mask=mask, other=0.0).to(tl.float32)
        term1 = tl.dot(B_T, dy_tile)
        term2 = tl.dot(dA_sym, x_tile)
        dx_tile = a_val * dy_tile + term1 + term2
        tl.store(ptrs_dx, dx_tile.to(tl.bfloat16), mask=mask)


def run_fused_ns(X, steps=1, return_norm=False):
    """Legacy wrapper for [B*L, N, D] layout."""
    orig_shape = X.shape
    BL = X.numel() // (X.shape[-1] * X.shape[-2])
    N, D = X.shape[-2], X.shape[-1]
    
    X_flat = X.reshape(BL, N, D)
    
    eps = 1e-7
    norm = X_flat.float().norm(dim=(1, 2), keepdim=True) + eps
    X_normed = (X_flat.float() / norm).to(X.dtype)
    
    Out = torch.empty_like(X_normed)
    BLOCK_D = 128 if D >= 128 else 64
    grid = (BL,)
    
    stride_batch = X_normed.stride(0)
    stride_n = X_normed.stride(1)
    stride_d = X_normed.stride(2)
    
    curr_X = X_normed
    for _ in range(steps):
        _fused_ns_step_kernel[grid](
            curr_X, Out,
            stride_batch, stride_n, stride_d,
            N=N, D=D, BLOCK_D=BLOCK_D
        )
        curr_X = Out
    
    if return_norm:
        return Out.reshape(orig_shape), X_normed, norm
    return Out.reshape(orig_shape)


def run_fused_ns_backward(X_normed, dY, norm, steps=1):
    """Legacy backward wrapper for [B*L, N, D] layout."""
    BL, N, D = X_normed.shape
    assert N == 16, "Kernel optimized for N=16"
    
    dX_normed = torch.empty_like(X_normed)
    grid = (BL,)
    BLOCK_D = 128
    
    _fused_ns_backward_kernel[grid](
        X_normed, dY, dX_normed,
        X_normed.stride(0), X_normed.stride(1), X_normed.stride(2),
        N=N, D=D, BLOCK_D=BLOCK_D
    )
    
    dX = (dX_normed.float() / norm).to(X_normed.dtype)
    return dX
