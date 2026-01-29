# Copyright (c) 2024, Momentum Mamba Team.
# Newton-Schulz orthogonalization using Triton + PyTorch hybrid

import torch
import triton
import triton.language as tl

from .matmul_transpose import matmul_transpose_assign


@triton.jit
def fused_norm_ns_kernel(
    # Input/output pointers
    G_ptr,           # Input matrix [B*L, D, N]
    Out_ptr,         # Output matrix [B*L, D, N]
    # Matrix dimensions
    BL,              # B * L (number of matrices)
    D: tl.constexpr, # d_inner (rows)
    N: tl.constexpr, # d_state (cols)
    # Strides for [BL, D, N] layout
    stride_m,        # stride for matrix dim
    stride_d,        # stride for D dim
    stride_n,        # stride for N dim
    # NS coefficients
    a: tl.constexpr,
    b: tl.constexpr,
    c: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused Newton-Schulz kernel for small matrices.
    Each program handles one [D, N] matrix.
    
    Assumes D <= BLOCK_D and N <= BLOCK_N (small matrices fit in one block).
    """
    pid = tl.program_id(0)
    if pid >= BL:
        return
    
    # Base offset for this matrix
    G_base = G_ptr + pid * stride_m
    Out_base = Out_ptr + pid * stride_m
    
    # Load entire matrix into registers
    d_offs = tl.arange(0, BLOCK_D)
    n_offs = tl.arange(0, BLOCK_N)
    
    d_mask = d_offs < D
    n_mask = n_offs < N
    mask_2d = d_mask[:, None] & n_mask[None, :]
    
    # Load X [D, N]
    X = tl.load(
        G_base + d_offs[:, None] * stride_d + n_offs[None, :] * stride_n,
        mask=mask_2d,
        other=0.0
    ).to(tl.float32)
    
    # Compute Frobenius norm
    X_sq = X * X
    norm_sq = tl.sum(X_sq)
    norm = tl.sqrt(norm_sq) + 1e-7
    
    # Normalize
    X = X / norm
    
    # Newton-Schulz iteration (single step)
    # A = X @ X.T  [D, N] @ [N, D] = [D, D]
    # Need to compute this carefully in blocks
    
    # For small D, N: use tl.dot
    # X: [BLOCK_D, BLOCK_N], X.T: [BLOCK_N, BLOCK_D]
    X_T = tl.trans(X)  # [BLOCK_N, BLOCK_D]
    
    # A = X @ X.T -> [BLOCK_D, BLOCK_D]
    A = tl.dot(X, X_T, allow_tf32=True)
    
    # AA = A @ A
    AA = tl.dot(A, A, allow_tf32=True)
    
    # B_mat = b*A + c*AA
    B_mat = b * A + c * AA
    
    # X = a*X + B_mat @ X
    BX = tl.dot(B_mat, X, allow_tf32=True)
    X = a * X + BX
    
    # Store result
    tl.store(
        Out_base + d_offs[:, None] * stride_d + n_offs[None, :] * stride_n,
        X,
        mask=mask_2d
    )


def newton_schulz_triton_fwd(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization using Triton (single step).
    
    Args:
        G: Input tensor of shape [B, L, D, N] where D is d_inner, N is d_state
        steps: Number of NS iterations (currently only 1 is supported efficiently)
        
    Returns:
        Orthogonalized tensor of shape [B, L, D, N]
    """
    assert G.ndim == 4, f"Expected 4D tensor [B, L, D, N], got {G.ndim}D"
    B, L, D, N = G.shape
    
    # For Triton, we need D and N to fit in blocks
    # Typical values: D ~ 32-256, N ~ 16
    BLOCK_D = triton.next_power_of_2(max(D, N))
    BLOCK_N = BLOCK_D  # Make square for matmul
    
    # Cap at reasonable size for registers
    if BLOCK_D > 64 or BLOCK_N > 64:
        # Fall back to PyTorch for large matrices
        return _newton_schulz_pytorch_fwd(G, steps)
    
    # NS coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Handle transpose case (D > N) in PyTorch
    transposed = D > N
    if transposed:
        G = G.mT.contiguous()  # [B, L, N, D]
        B, L, D, N = G.shape  # Now D=N_orig, N=D_orig
    
    # Reshape to [B*L, D, N] for kernel
    G_flat = G.reshape(B * L, D, N).contiguous()
    Out_flat = torch.empty_like(G_flat)
    
    # Launch kernel
    grid = (B * L,)
    
    # Handle multiple steps by calling kernel multiple times
    for step in range(steps):
        if step == 0:
            input_tensor = G_flat
        else:
            input_tensor = Out_flat.clone()
        
        fused_norm_ns_kernel[grid](
            input_tensor, Out_flat,
            B * L, D, N,
            G_flat.stride(0), G_flat.stride(1), G_flat.stride(2),
            a, b, c,
            BLOCK_D, BLOCK_N,
        )
    
    # Reshape back
    Out = Out_flat.reshape(B, L, D, N)
    
    if transposed:
        Out = Out.mT.contiguous()
    
    return Out


def _newton_schulz_pytorch_fwd(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """Fallback PyTorch implementation for large matrices."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    
    # Normalize
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm + 1e-7)
    
    # NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B_mat = b * A + c * A @ A
        X = a * X + B_mat @ X
    
    if transposed:
        X = X.mT
    
    return X


class NewtonSchulzTriton(torch.autograd.Function):
    """
    Newton-Schulz orthogonalization with exact gradients.
    Forward uses Triton (when possible), backward uses PyTorch autograd.
    """
    
    @staticmethod
    def forward(ctx, G: torch.Tensor, steps: int = 1):
        ctx.save_for_backward(G)
        ctx.steps = steps
        
        # Use Triton kernel
        Out = newton_schulz_triton_fwd(G, steps)
        return Out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        G, = ctx.saved_tensors
        steps = ctx.steps
        
        # Recompute forward with autograd for exact gradients
        G_grad = G.detach().clone().requires_grad_(True)
        
        with torch.enable_grad():
            a, b, c = 3.4445, -4.7750, 2.0315
            X = G_grad.float()
            
            transposed = G_grad.size(-2) > G_grad.size(-1)
            if transposed:
                X = X.mT
            
            # Normalize (differentiable)
            norm = X.norm(dim=(-2, -1), keepdim=True)
            X = X / (norm + 1e-7)
            
            # NS iterations (differentiable)
            for _ in range(steps):
                A = X @ X.mT
                B_mat = b * A + c * A @ A
                X = a * X + B_mat @ X
            
            if transposed:
                X = X.mT
            
            # Compute gradient
            X.backward(grad_output)
        
        return G_grad.grad, None


def newton_schulz_triton(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization using Triton with autograd support.
    
    Args:
        G: Input tensor of shape [B, L, D, N]
        steps: Number of NS iterations
        
    Returns:
        Orthogonalized tensor of shape [B, L, D, N]
    """
    return NewtonSchulzTriton.apply(G, steps)


# =============================================================================
# Flash Muon Style Implementation (optimized matmul_transpose)
# =============================================================================

def fast_newtonschulz_2d(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    Fast Newton-Schulz for 2D matrices using optimized matmul_transpose.
    Adapted from flash_muon.
    
    Args:
        G: Input tensor [M, K] (2D matrix)
        steps: Number of NS iterations
        
    Returns:
        Orthogonalized tensor [M, K]
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT.contiguous()
    
    M, K = X.shape
    
    # Pre-allocate buffers for A = X @ X.T and AA = A @ A
    buf1 = torch.empty(M, M, dtype=X.dtype, device=X.device)
    buf2 = torch.empty(M, M, dtype=X.dtype, device=X.device)
    
    # Normalize
    X = X / (X.norm() + 1e-7)
    
    # NS iterations with optimized matmul_transpose
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)        # buf1 = X @ X.T (A)
        matmul_transpose_assign(buf1, buf2)     # buf2 = A @ A
        B = b * buf1 + c * buf2
        X = a * X + B @ X
    
    if transposed:
        X = X.mT.contiguous()
    
    return X


def fast_newtonschulz_batched(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    Fast Newton-Schulz for batched 4D tensors [B, L, D, N].
    Uses flash_muon's optimized matmul_transpose for each (B, L) matrix.
    
    Args:
        G: Input tensor [B, L, D, N]
        steps: Number of NS iterations
        
    Returns:
        Orthogonalized tensor [B, L, D, N]
    """
    assert G.ndim == 4
    B, L, D, N = G.shape
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Work in bfloat16 for speed
    X = G.bfloat16()
    
    # Handle transpose case
    transposed = D > N
    if transposed:
        X = X.mT.contiguous()  # [B, L, N, D]
        B, L, D, N = X.shape
    
    # Reshape to process all B*L matrices
    X = X.reshape(B * L, D, N)
    
    # Pre-allocate buffers (one set per matrix in batch)
    # For efficiency, process matrices one at a time with buffer reuse
    buf1 = torch.empty(D, D, dtype=X.dtype, device=X.device)
    buf2 = torch.empty(D, D, dtype=X.dtype, device=X.device)
    
    # Normalize all matrices
    norms = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norms + 1e-7)
    
    # Process each matrix (can't easily batch the custom kernel)
    # For large batches, fall back to standard PyTorch
    if B * L > 64:
        # Use vectorized PyTorch for large batches
        for _ in range(steps):
            A = X @ X.mT
            AA = A @ A
            B_mat = b * A + c * AA
            X = a * X + B_mat @ X
    else:
        # Use optimized kernel for small batches
        results = []
        for i in range(B * L):
            Xi = X[i].contiguous()
            for _ in range(steps):
                matmul_transpose_assign(Xi, buf1)
                matmul_transpose_assign(buf1, buf2)
                B_mat = b * buf1 + c * buf2
                Xi = a * Xi + B_mat @ Xi
            results.append(Xi)
        X = torch.stack(results)
    
    X = X.reshape(B, L, D, N)
    
    if transposed:
        X = X.mT.contiguous()
    
    return X


class FastNewtonSchulzTriton(torch.autograd.Function):
    """
    Fast Newton-Schulz using flash_muon's optimized matmul_transpose.
    Forward uses optimized Triton kernel, backward uses PyTorch autograd.
    """
    
    @staticmethod
    def forward(ctx, G: torch.Tensor, steps: int = 1):
        ctx.save_for_backward(G)
        ctx.steps = steps
        
        Out = fast_newtonschulz_batched(G, steps)
        return Out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        G, = ctx.saved_tensors
        steps = ctx.steps
        
        # Recompute forward with autograd for exact gradients
        G_grad = G.detach().clone().requires_grad_(True)
        
        with torch.enable_grad():
            a, b, c = 3.4445, -4.7750, 2.0315
            X = G_grad.float()
            
            transposed = G_grad.size(-2) > G_grad.size(-1)
            if transposed:
                X = X.mT
            
            norm = X.norm(dim=(-2, -1), keepdim=True)
            X = X / (norm + 1e-7)
            
            for _ in range(steps):
                A = X @ X.mT
                B_mat = b * A + c * A @ A
                X = a * X + B_mat @ X
            
            if transposed:
                X = X.mT
            
            X.backward(grad_output)
        
        return G_grad.grad, None


def fast_newton_schulz_triton(G: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """
    Fast Newton-Schulz using flash_muon's optimized matmul_transpose with autograd.
    
    Args:
        G: Input tensor of shape [B, L, D, N]
        steps: Number of NS iterations
        
    Returns:
        Orthogonalized tensor of shape [B, L, D, N]
    """
    return FastNewtonSchulzTriton.apply(G, steps)
