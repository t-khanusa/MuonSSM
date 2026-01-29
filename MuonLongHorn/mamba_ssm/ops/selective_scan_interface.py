# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat, einsum, reduce

import fast_layer_norm
try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

import selective_scan_cuda
from mamba_ssm.ops.triton.layernorm import _layer_norm_fwd, _layer_norm_bwd

# Import Triton NS for optimized Newton-Schulz
try:
    from mamba_ssm.ops.triton.newton_schulz_triton import (
        newton_schulz_triton,
        newton_schulz_fwd_bwd_combined,
        _newton_schulz_batched_forward,
        _newton_schulz_batched_backward,
    )
    TRITON_NS_AVAILABLE = True
except ImportError:
    TRITON_NS_AVAILABLE = False

# Import fused Triton NS kernel (true Triton implementation)
try:
    from mamba_ssm.ops.triton.ns_triton import (
        run_fused_ns, run_fused_ns_backward,  # Legacy [B*L, N, D] layout
        run_fused_ns_native, run_fused_ns_native_backward,  # Native [B, L, D, N] layout
    )
    FUSED_TRITON_NS_AVAILABLE = True
except ImportError:
    FUSED_TRITON_NS_AVAILABLE = False


@torch.jit.script
def get_dk(dTK: torch.Tensor, dK: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return dK + dTK.sum(1, keepdim=True).unsqueeze(1) * 2.0 * K


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)







def selective_scan_online7_fn(u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False,
                               beta=0.0, alpha=1.0):
    """Original Longhorn selective scan with optional momentum (no Newton-Schulz)."""
    return SelectiveScanOnline7Fn.apply(u, Q, K, T, D, t_bias, z, return_last_state, beta, alpha)


def selective_scan_online_orth_fn(u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False,
                                   beta=0.9, alpha=1.0, ns_steps=1, ns_mode='compile'):
    """MuonLonghorn: Selective scan with Newton-Schulz orthogonalization and momentum.
    
    Args:
        u: Input tensor [B, D, L]
        Q: Query tensor [B, G, N, L] or [B, N, L]
        K: Key tensor [B, G, N, L] or [B, N, L]
        T: Time step tensor [B, D, L]
        D: Skip connection weight [D]
        t_bias: Time step bias [D]
        z: Gating tensor [B, D, L]
        return_last_state: Whether to return last hidden state
        beta: Velocity decay factor (momentum)
        alpha: Velocity scale factor
        ns_steps: Number of Newton-Schulz iterations
        ns_mode: 'compile' for torch.compile or 'triton' for Triton kernel
    
    Returns:
        Output tensor [B, D, L], optionally with last_state
    """
    return SelectiveScanOnlineOrthFn.apply(u, Q, K, T, D, t_bias, z, return_last_state, 
                                            beta, alpha, ns_steps, ns_mode)


class SelectiveScanOnline7Fn(torch.autograd.Function):
    """Original Longhorn selective scan with optional momentum (no Newton-Schulz).
    
    Class Variables for Analysis (shared with SelectiveScanOnlineOrthFn):
        Uses SelectiveScanOnlineOrthFn's class variables for tensor capture.
    """

    @staticmethod
    def forward(ctx, u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False,
                beta=0.0, alpha=1.0):
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        if K.dim() == 3:
            K = rearrange(K, "b dstate l -> b 1 dstate l")
            ctx.squeeze_K = True
        if Q.dim() == 3:
            Q = rearrange(Q, "b dstate l -> b 1 dstate l")
            ctx.squeeze_Q = True
        
        # Capture tensors for analysis if enabled (no NS version)
        if SelectiveScanOnlineOrthFn._capture_enabled:
            B, G, N, L = K.shape
            D_dim = u.shape[1]
            
            T_compute = T
            if t_bias is not None:
                T_compute = T + t_bias.unsqueeze(-1)
            dt = torch.sigmoid(T_compute)
            
            K2_sum = (K ** 2).sum(dim=2, keepdim=True)
            K2_sum = K2_sum.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, 1, L)
            dt_norm = dt / (1 + dt * K2_sum.squeeze(2))
            
            K_expanded = K.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, N, L)
            deltaB_u = torch.einsum('bdl,bdnl->bdln', dt_norm * u, K_expanded)
            
            # Store as "before NS" (no NS applied in this version)
            SelectiveScanOnlineOrthFn._last_deltaB_u = deltaB_u.detach().clone()
            # For no-NS case, "after NS" is just the normalized version
            deltaB_u_for_ns = deltaB_u.permute(0, 2, 1, 3)
            X_normed = deltaB_u_for_ns.float() / (deltaB_u_for_ns.float().norm(dim=(-2, -1), keepdim=True) + 1e-7)
            SelectiveScanOnlineOrthFn._last_deltaB_u_orth = X_normed.permute(0, 2, 1, 3).detach().clone()
        
        import online_selective_scan_cuda
        out, x, *rest = online_selective_scan_cuda.online_fwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
            beta,  # MuonLonghorn: momentum parameters
            alpha,
        )

        ctx.has_z = z is not None
        ctx.return_last_state = return_last_state
        ctx.beta = beta
        ctx.alpha = alpha
        # MuonLonghorn: state indices depend on momentum usage
        use_momentum = beta > 0.0
        last_state = x[:, :, -1, 1::2] if not use_momentum else x[:, :, -1, 3::4]  # (batch, dim, dstate)
        ctx.save_for_backward(u, Q, K, T, D, t_bias, x, z, out)
        if not ctx.has_z:
            return out if not return_last_state else (out, last_state)
        else:
            out = rest[0]
            return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.return_last_state:
            raise NotImplementedError(
                "backward with return_last_state is not implemented"
            )
        u, Q, K, T, D, t_bias, x, z, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        import online_selective_scan_cuda
        du, dQ, dK, dT, dD, dz, dt_bias, dTK = online_selective_scan_cuda.online_bwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
            dout,
            x,
            out,
            None,
            ctx.beta,   # MuonLonghorn: momentum parameters
            ctx.alpha,
        )
        dK = get_dk(dTK, dK, K).to(K)

        dK = dK.squeeze(1) if getattr(ctx, "squeeze_K", False) else dK
        dQ = dQ.squeeze(1) if getattr(ctx, "squeeze_Q", False) else dQ
        return du, dQ, dK, dT, dD if D is not None else None, dt_bias, dz if z is not None else None, None, None, None


class SelectiveScanOnlineOrthFn(torch.autograd.Function):
    """MuonLonghorn: Selective scan with Newton-Schulz orthogonalization and momentum.
    
    Forward:
        1. Compute velocity input: input_t = dt_norm * u * K (where dt_norm = sigmoid(T + t_bias) / (1 + dt * K^2))
        2. Apply Newton-Schulz orthogonalization to input_t
        3. Call CUDA with orthogonalized input, beta, alpha for momentum scan
        
    Backward:
        1. Call CUDA backward to get d_deltaB_u_orth (gradient w.r.t. orthogonalized input)
        2. Backpropagate through Newton-Schulz using autograd
        3. Compute gradients for u, K, T
    
    Class Variables for Analysis:
        _last_deltaB_u: Last velocity input tensor before NS [B, D, L, N]
        _last_deltaB_u_orth: Last velocity input tensor after NS [B, D, L, N]
        _capture_enabled: Set to True to enable tensor capture (disabled by default for performance)
    """
    
    # Class variables for tensor analysis (disabled by default)
    _capture_enabled = False
    _last_deltaB_u = None
    _last_deltaB_u_orth = None
    
    @classmethod
    def enable_capture(cls, enabled=True):
        """Enable/disable tensor capture for analysis."""
        cls._capture_enabled = enabled
        if not enabled:
            cls._last_deltaB_u = None
            cls._last_deltaB_u_orth = None
    
    @classmethod
    def get_captured_tensors(cls):
        """Get the last captured tensors."""
        return cls._last_deltaB_u, cls._last_deltaB_u_orth
    
    @staticmethod
    def forward(ctx, u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False,
                beta=0.9, alpha=1.0, ns_steps=1, ns_mode='ns_compile'):
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        if K.dim() == 3:
            K = rearrange(K, "b dstate l -> b 1 dstate l")
            ctx.squeeze_K = True
        if Q.dim() == 3:
            Q = rearrange(Q, "b dstate l -> b 1 dstate l")
            ctx.squeeze_Q = True
        
        B, G, N, L = K.shape
        D_dim = u.shape[1]
        
        # Step 1: Compute velocity input (before NS) - keep original dtype
        # T_sigmoid = sigmoid(T + t_bias) for dt
        T_compute = T
        if t_bias is not None:
            T_compute = T + t_bias.unsqueeze(-1)
        dt = torch.sigmoid(T_compute)
        
        # K^2 sum for dt normalization
        K2_sum = (K ** 2).sum(dim=2, keepdim=True)  # [B, G, 1, L]
        K2_sum = K2_sum.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, 1, L)  # [B, D, 1, L]
        dt_norm = dt / (1 + dt * K2_sum.squeeze(2))  # [B, D, L]
        
        # velocity input = dt_norm * u * K  [B, D, L] * [B, D, L] * [B, G, N, L]
        # Expand K to match D dimension
        K_expanded = K.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, N, L)  # [B, D, N, L]
        
        # deltaB_u = dt_norm * u for each state dimension
        # Shape: [B, D, L, N] for NS (treating L as "sequence" for per-timestep orthogonalization)
        deltaB_u = torch.einsum('bdl,bdnl->bdln', dt_norm * u, K_expanded)  # [B, D, L, N]
        
        # Step 2: Apply Newton-Schulz orthogonalization
        # NS operates on last two dims, so permute [B, D, L, N] -> [B, L, D, N]
        # NS will batch over [B, L] and orthogonalize [D, N] matrices
        deltaB_u_for_ns = deltaB_u.permute(0, 2, 1, 3).contiguous()  # [B, L, D, N]
        
        # Run NS based on mode
        if ns_mode == 'triton_native' and FUSED_TRITON_NS_AVAILABLE:
            # NATIVE LAYOUT: Works directly on [B, L, D, N] - NO RESHAPE!
            with torch.no_grad():
                deltaB_u_orth = run_fused_ns_native(deltaB_u_for_ns, steps=ns_steps)
        elif ns_mode == 'triton_fused' and FUSED_TRITON_NS_AVAILABLE:
            # Use true Triton kernel (fused operations, register-resident A/B)
            # Input: [B, L, D, N] -> need [B*L, N, D] for kernel
            B_ns, L_ns, D_ns, N_ns = deltaB_u_for_ns.shape
            # Permute to [B, L, N, D] then reshape to [B*L, N, D]
            deltaB_u_ns_input = deltaB_u_for_ns.permute(0, 1, 3, 2).reshape(B_ns * L_ns, N_ns, D_ns)
            with torch.no_grad():
                deltaB_u_ns_out = run_fused_ns(deltaB_u_ns_input, steps=ns_steps)
            # Reshape back to [B, L, N, D] then permute to [B, L, D, N]
            deltaB_u_orth = deltaB_u_ns_out.reshape(B_ns, L_ns, N_ns, D_ns).permute(0, 1, 3, 2)
        elif ns_mode == 'triton' and TRITON_NS_AVAILABLE:
            # Use optimized Triton NS (bmm-based)
            with torch.no_grad():
                deltaB_u_orth = newton_schulz_triton(deltaB_u_for_ns, steps=ns_steps)
        else:
            # Use PyTorch NS (default/compile mode)
            with torch.no_grad():
                deltaB_u_orth = _newton_schulz_pytorch(deltaB_u_for_ns, steps=ns_steps)
        
        # Permute back to [B, D, L, N] and convert to float32 for CUDA kernel
        deltaB_u_orth = deltaB_u_orth.float().permute(0, 2, 1, 3).contiguous()
        
        # Capture tensors for analysis if enabled
        if SelectiveScanOnlineOrthFn._capture_enabled:
            SelectiveScanOnlineOrthFn._last_deltaB_u = deltaB_u.detach().clone()
            SelectiveScanOnlineOrthFn._last_deltaB_u_orth = deltaB_u_orth.detach().clone()
        
        # Step 3: Call CUDA with orthogonalized input
        import online_selective_scan_cuda
        out, x, *rest = online_selective_scan_cuda.online_fwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
            beta,
            alpha,
            deltaB_u_orth,  # orthogonalized velocity input
        )

        ctx.has_z = z is not None
        ctx.return_last_state = return_last_state
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.ns_steps = ns_steps
        ctx.ns_mode = ns_mode
        
        # MuonLonghorn: state indices - with momentum, states are [vel, hidden] pairs
        last_state = x[:, :, -1, 3::4]  # hidden state at odd+1 positions
        
        # MEMORY OPTIMIZATION: Don't save deltaB_u and deltaB_u_orth (576 MB per layer!)
        # Instead, recompute them in backward pass from saved u, Q, K, T, t_bias
        # All intermediate values (dt_norm, K_expanded) can be recomputed from saved tensors
        ctx.save_for_backward(u, Q, K, T, D, t_bias, x, z, out)
        if not ctx.has_z:
            return out if not return_last_state else (out, last_state)
        else:
            out = rest[0]
            return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.return_last_state:
            raise NotImplementedError(
                "backward with return_last_state is not implemented"
            )
        # MEMORY OPTIMIZATION: Recompute deltaB_u and deltaB_u_orth instead of loading from saved tensors
        u, Q, K, T, D, t_bias, x, z, out = ctx.saved_tensors
        
        # Store original dtype
        orig_dtype = u.dtype
        
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # RECOMPUTE deltaB_u and deltaB_u_orth (instead of loading from saved tensors)
        B, G, N, L = K.shape
        D_dim = u.shape[1]
        
        # Recompute dt_norm using ORIGINAL dtype (no float32 conversion)
        T_compute = T
        if t_bias is not None:
            T_compute = T + t_bias.unsqueeze(-1)
        dt = torch.sigmoid(T_compute)  # [B, D, L]
        K2_sum = (K ** 2).sum(dim=2, keepdim=True)  # [B, G, 1, L]
        K2_sum_expanded = K2_sum.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, 1, L).squeeze(2)  # [B, D, L]
        denominator = 1 + dt * K2_sum_expanded  # [B, D, L]
        dt_norm = dt / denominator  # [B, D, L]
        
        # Recompute K_expanded
        K_expanded = K.expand(-1, D_dim // G, -1, -1).reshape(B, D_dim, N, L)
        
        # RECOMPUTE deltaB_u (same as forward pass)
        deltaB_u = torch.einsum('bdl,bdnl->bdln', dt_norm * u, K_expanded)  # [B, D, L, N]
        B_size, D_dim_size, L_size, N_size = deltaB_u.shape
        
        # NS operates on last two dims, so permute [B, D, L, N] -> [B, L, D, N]
        deltaB_u_for_ns = deltaB_u.permute(0, 2, 1, 3).contiguous()  # [B, L, D, N]
        del deltaB_u
        
        # Use mode-specific NS backward
        use_triton_bwd = ctx.ns_mode == 'triton' and TRITON_NS_AVAILABLE
        use_triton_fused_bwd = ctx.ns_mode == 'triton_fused' and FUSED_TRITON_NS_AVAILABLE
        use_triton_native_bwd = ctx.ns_mode == 'triton_native' and FUSED_TRITON_NS_AVAILABLE
        
        if use_triton_native_bwd:
            # NATIVE LAYOUT: Works directly on [B, L, D, N] - NO RESHAPE!
            with torch.no_grad():
                deltaB_u_orth_result, X_normed_native, norm_native = run_fused_ns_native(
                    deltaB_u_for_ns, steps=ctx.ns_steps, return_intermediates=True
                )
            # Permute for CUDA: [B, L, D, N] -> [B, D, L, N]
            deltaB_u_orth = deltaB_u_orth_result.float().permute(0, 2, 1, 3).contiguous()
        elif use_triton_fused_bwd:
            # FUSED TRITON: Use true Triton kernel for forward (need X_normed, norm for backward)
            B_ns, L_ns, D_ns, N_ns = deltaB_u_for_ns.shape
            # Permute to [B, L, N, D] then reshape to [B*L, N, D]
            deltaB_u_ns_input = deltaB_u_for_ns.permute(0, 1, 3, 2).reshape(B_ns * L_ns, N_ns, D_ns).contiguous()
            with torch.no_grad():
                # return_norm=True to get X_normed and norm for backward
                deltaB_u_ns_out, X_normed_fused, norm_fused = run_fused_ns(
                    deltaB_u_ns_input, steps=ctx.ns_steps, return_norm=True
                )
            # Reshape back to [B, L, N, D] then permute to [B, L, D, N]
            deltaB_u_orth_result = deltaB_u_ns_out.reshape(B_ns, L_ns, N_ns, D_ns).permute(0, 1, 3, 2)
            # Permute for CUDA: [B, L, D, N] -> [B, D, L, N]
            deltaB_u_orth = deltaB_u_orth_result.float().permute(0, 2, 1, 3).contiguous()
        elif use_triton_bwd:
            # OPTIMIZED: Use Triton direct forward+backward (no autograd overhead)
            deltaB_u_orth_result, X_normed, norm, transposed = _newton_schulz_batched_forward(
                deltaB_u_for_ns, steps=ctx.ns_steps
            )
            # Permute for CUDA: [B, L, D, N] -> [B, D, L, N]
            deltaB_u_orth = deltaB_u_orth_result.float().permute(0, 2, 1, 3).contiguous()
        else:
            # ORIGINAL: Run NS with grad tracking for gradient computation
            deltaB_u_for_ns_grad = deltaB_u_for_ns.clone().requires_grad_(True)
            
            with torch.enable_grad():
                deltaB_u_orth_ns = _newton_schulz_pytorch(deltaB_u_for_ns_grad, steps=ctx.ns_steps)
            
            # Permute for CUDA: [B, L, D, N] -> [B, D, L, N]
            deltaB_u_orth = deltaB_u_orth_ns.detach().float().permute(0, 2, 1, 3).contiguous()

        import online_selective_scan_cuda
        # Call CUDA backward - returns gradient w.r.t. orthogonalized input
        results = online_selective_scan_cuda.online_bwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
            dout,
            x,
            out,
            None,
            ctx.beta,
            ctx.alpha,
            deltaB_u_orth,  # pass recomputed orthogonalized input
        )
        
        # Free deltaB_u_orth immediately after use
        del deltaB_u_orth
        
        # Results: du, dQ, dK, dT, dD, dz, dt_bias, dTK, d_deltaB_u_orth
        du, dQ, dK, dT, dD, dz, dt_bias_out, dTK, d_deltaB_u_orth = results
        
        # d_deltaB_u_orth is [B, D, L, N], permute to match NS output [B, L, D, N]
        d_deltaB_u_orth_permuted = d_deltaB_u_orth.permute(0, 2, 1, 3).contiguous()  # [B, L, D, N]
        del d_deltaB_u_orth
        
        if use_triton_native_bwd:
            # NATIVE LAYOUT: Direct backward on [B, L, D, N] - NO RESHAPE!
            dX_native = run_fused_ns_native_backward(
                X_normed_native, d_deltaB_u_orth_permuted, norm_native, steps=ctx.ns_steps
            )
            del X_normed_native, norm_native, d_deltaB_u_orth_permuted
            # Permute: [B, L, D, N] -> [B, D, L, N]
            d_deltaB_u = dX_native.float().permute(0, 2, 1, 3).contiguous()
            del dX_native
        elif use_triton_fused_bwd:
            # FUSED TRITON: Use true Triton kernel for backward
            # d_deltaB_u_orth_permuted: [B, L, D, N] -> need [B*L, N, D]
            B_ns, L_ns, D_ns, N_ns = d_deltaB_u_orth_permuted.shape
            dY_fused = d_deltaB_u_orth_permuted.permute(0, 1, 3, 2).reshape(B_ns * L_ns, N_ns, D_ns).contiguous()
            # Call fused backward with X_normed and norm from forward
            # X_normed_fused: [B*L, N, D] (already flat), norm_fused: [B*L, 1, 1]
            dX_fused = run_fused_ns_backward(X_normed_fused, dY_fused, norm_fused, steps=ctx.ns_steps)
            del dY_fused, X_normed_fused, norm_fused, deltaB_u_ns_input
            # Reshape back: [B*L, N, D] -> [B, L, N, D] -> [B, D, L, N]
            d_deltaB_u = dX_fused.reshape(B_ns, L_ns, N_ns, D_ns).permute(0, 3, 1, 2).float().contiguous()
            del dX_fused, d_deltaB_u_orth_permuted
        elif use_triton_bwd:
            # OPTIMIZED: Direct backward computation without autograd
            d_deltaB_u_ns = _newton_schulz_batched_backward(
                d_deltaB_u_orth_permuted, X_normed, norm, transposed, ctx.ns_steps
            )
            del d_deltaB_u_orth_permuted, X_normed, norm
            # Permute: [B, L, D, N] -> [B, D, L, N] and ensure float32
            d_deltaB_u = d_deltaB_u_ns.float().permute(0, 2, 1, 3).contiguous()
            del d_deltaB_u_ns
        else:
            # ORIGINAL: Backward through NS using the already-computed graph
            deltaB_u_orth_ns.backward(d_deltaB_u_orth_permuted)
            del deltaB_u_orth_ns, d_deltaB_u_orth_permuted
            
            # Get gradient: [B, L, D, N] -> [B, D, L, N]
            d_deltaB_u = deltaB_u_for_ns_grad.grad.permute(0, 2, 1, 3).contiguous()
            del deltaB_u_for_ns_grad
        
        del deltaB_u_for_ns
        
        # Compute gradients - d_deltaB_u is float32, so convert K_expanded/dt_norm/u to match
        # d(dt_norm * u * K) / d(dt_norm) = u * K
        d_dt_norm_u = torch.einsum('bdln,bdnl->bdl', d_deltaB_u, K_expanded.float())  # gradient w.r.t. (dt_norm * u)
        
        # du contribution from orthogonalized path: d(dt_norm * u)/du = dt_norm
        du_orth = d_dt_norm_u * dt_norm.float()
        
        # dK contribution from orthogonalized path
        dK_orth = torch.einsum('bdln,bdl->bdnl', d_deltaB_u, (dt_norm * u).float())
        dK_orth = dK_orth.reshape(B, G, D_dim // G, N, L).sum(dim=2)  # [B, G, N, L]
        
        # Free d_deltaB_u after use
        del d_deltaB_u
        
        # dT contribution from orthogonalized path
        d_dt_norm = d_dt_norm_u * u.float()  # gradient w.r.t. dt_norm itself [B, D, L]
        del d_dt_norm_u
        
        d_dt_norm_d_dt = 1.0 / (denominator.float() ** 2)  # [B, D, L]
        d_dt_d_T = dt.float() * (1 - dt.float())  # sigmoid derivative [B, D, L]
        dT_orth = d_dt_norm * d_dt_norm_d_dt * d_dt_d_T  # [B, D, L]
        
        # Free intermediates
        del d_dt_norm, d_dt_norm_d_dt, d_dt_d_T, dt, denominator, dt_norm, K_expanded
        
        # Combine with du, dK, dT from CUDA backward (which handles forget gate path)
        du = du + du_orth.to(orig_dtype)
        dK = dK + dK_orth.to(orig_dtype)
        dT = dT + dT_orth.to(orig_dtype)
        
        del du_orth, dK_orth, dT_orth
        
        dK = get_dk(dTK, dK, K).to(orig_dtype)

        dK = dK.squeeze(1) if getattr(ctx, "squeeze_K", False) else dK
        dQ = dQ.squeeze(1) if getattr(ctx, "squeeze_Q", False) else dQ
        return du, dQ, dK, dT, dD if D is not None else None, dt_bias_out, dz if z is not None else None, None, None, None, None, None


class NewtonSchulzFunction(torch.autograd.Function):
    """Memory-efficient Newton-Schulz with recomputation in backward.
    
    Instead of saving intermediate tensors (A, B) for backward, we only save
    the normalized input X_normed. During backward, we recompute the forward
    pass to get gradients. This trades compute for memory.
    
    For 1 NS step: saves ~300 MB per call by not storing A, B, B@X intermediates.
    """
    
    @staticmethod
    def forward(ctx, G, steps=1):
        a, b, c = 3.4445, -4.7750, 2.0315
        
        # Convert to bf16
        X = G.bfloat16()
        
        # Transpose if rows > cols (work with smaller dim as rows for efficiency)
        transposed = G.size(-2) > G.size(-1)
        if transposed:
            X = X.transpose(-2, -1).contiguous()
        
        # Normalize
        norm = X.norm(dim=(-2, -1), keepdim=True) + 1e-7
        X_normed = X / norm
        
        # NS iterations (no grad tracking here)
        X = X_normed
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if transposed:
            X = X.transpose(-2, -1).contiguous()
            X_normed = X_normed.transpose(-2, -1).contiguous()
        
        # Save only what we need for backward: normalized input and metadata
        # Detach to prevent autograd from tracking these (we handle gradients manually)
        ctx.save_for_backward(X_normed.detach(), norm.detach())
        ctx.transposed = transposed
        ctx.steps = steps
        ctx.input_shape = G.shape
        
        return X
    
    @staticmethod
    def backward(ctx, dY):
        a, b, c = 3.4445, -4.7750, 2.0315
        X_normed, norm = ctx.saved_tensors
        transposed = ctx.transposed
        steps = ctx.steps
        
        # Handle transpose for backward - use explicit transpose and make contiguous
        if transposed:
            dY = dY.transpose(-2, -1).contiguous()
            X_normed = X_normed.transpose(-2, -1).contiguous()
        
        # Recompute forward pass to get intermediate X values for each step
        # We need X at each step to compute gradients
        X_list = [X_normed]
        X = X_normed
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * A @ A
            X = a * X + B @ X
            if _ < steps - 1:  # Don't save last one
                X_list.append(X)
        
        # Backward through NS steps (reverse order)
        dX = dY.bfloat16().contiguous()
        
        for step in range(steps - 1, -1, -1):
            X = X_list[step]
            
            # Recompute A and B (small matrices on last dims)
            X_T = X.transpose(-2, -1)
            A = X @ X_T
            A_sq = A @ A
            B = b * A + c * A_sq
            del A_sq
            
            # dL/dB = dX @ X.T
            dB = dX @ X_T
            del X_T
            
            # dL/dA = b * dB + c * (dB @ A.T + A.T @ dB)
            A_T = A.transpose(-2, -1)
            dB_A = dB @ A_T
            dA = b * dB + c * (dB_A + A_T @ dB)
            del dB_A, dB, A_T
            
            # Make dA symmetric
            dA_sym = dA + dA.transpose(-2, -1)
            del dA, A
            
            # Compute: dX_new = a * dX + B.T @ dX + dA_sym @ X
            # Use torch.matmul which handles arbitrary batch dimensions
            B_T = B.transpose(-2, -1)
            dX = a * dX + B_T @ dX + dA_sym @ X
            del B, B_T, dA_sym
        
        # Gradient through normalization: X_normed = X / norm
        # dL/dX_input = dL/dX_normed / norm - X_normed * (dL/dX_normed * X_normed).sum() / norm
        dX_input = (dX - X_normed * (dX * X_normed).sum(dim=(-2, -1), keepdim=True)) / norm
        
        if transposed:
            dX_input = dX_input.transpose(-2, -1).contiguous()
        
        return dX_input, None  # None for steps


def _newton_schulz_pytorch(G: torch.Tensor, steps: int = 1, use_fp32: bool = False) -> torch.Tensor:
    """Newton-Schulz iteration to compute orthogonalization of G.
    
    Memory-efficient version that uses custom backward to avoid saving intermediates.
    Based on the official Muon implementation. Uses bf16 throughout for stability.
    
    Args:
        G: Input tensor of shape [..., D, N] (batched matrices)
        steps: Number of NS iterations (default 1)
        use_fp32: Ignored, kept for API compatibility (always uses bf16)
        
    Returns:
        Orthogonalized tensor of same shape as G
    """
    if G.requires_grad:
        return NewtonSchulzFunction.apply(G, steps)
    else:
        # Fast path without autograd overhead
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16()
        
        transposed = G.size(-2) > G.size(-1)
        if transposed:
            X = X.transpose(-2, -1).contiguous()
        
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if transposed:
            X = X.transpose(-2, -1).contiguous()
        
        return X


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_online7_ref(u, q, k, dt, D=None, t_bias=None, z=None, return_last_state=False, eps=1e-6):
    """
    To Rui:
        we will always use real numbers, so ignore all complex logic in mamba cuda code.
    """

    """
    u:  r(B D L)
    A:  r(D N) in [0, 1], should be very close to 1
    B:  r(D N) in [0, 1], should be very close to 1
    q:  r(B N L)
    k:  r(B N L)
    dt: r(B D L), also in [0, 1]
    forget_bias: r(1) a scalar > 0

    D: r(D)
    z: r(B D L)

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    q = q.float()
    k = k.float()
    dt = dt.float()
    if t_bias is not None:
        dt = dt + t_bias.float()[..., None]

    dt = torch.sigmoid(dt)
    dt = dt / (1 + dt * k.square().sum(dim=-2, keepdim=True))

    K = rearrange(k, 'b n l -> b 1 l n').pow(2)

    forget_gate = (1 - dt.unsqueeze(-1) * K)

    input_matrix = torch.einsum('bdl,bnl->bdln', (dt*u), k)

    last_state = None
    batch, dim = u.shape[:2]
    dstate = q.shape[1]

    x = q.new_zeros((batch, dim, dstate))
    ys = []
    for i in range(u.shape[2]):
        x = forget_gate[:, :, i] * x + input_matrix[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, q[:, :, i])

        if i == x.shape[2] - 1:
            last_state = x

        ys.append(y)

    y = torch.stack(ys, dim=-1)  # (batch dim L)

    out = y if D is None else y + u * rearrange(D, "d -> d 1")

    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


###############################################################################
#
# Mamba Inner Functions
#
###############################################################################


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    # @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    # @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


###############################################################################
#
# Longhorn Inner Function
#
###############################################################################


class LonghornInnerFn(torch.autograd.Function):

    @staticmethod
    # @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                norm_weight, out_proj_weight, D=None, delta_bias=None):
        """
             x: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        L = xz.shape[-1]
        R = delta_proj_weight.shape[1]
        DD = (x_proj_weight.shape[0] - R) // 2

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())

        if xz.stride(-1) != 1:
            xz = xz.contiguous()

        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :R].t(), "d (b l) -> b d l", l = L)

        K = x_dbl[:, R:R+DD]  # (bl dstate)
        Q = x_dbl[:, -DD:]  # (bl dstate)
        K = rearrange(K, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        Q = rearrange(Q, "(b l) dstate -> b 1 dstate l", l=L).contiguous()

        if D is not None:
            D = D.contiguous()

        import online_selective_scan_cuda
        if norm_weight is None:
            online_z = z
        else:
            online_z = None
        out, scan_intermediates, *rest = online_selective_scan_cuda.online_fwd(
            conv1d_out,
            Q,
            K,
            delta,
            D,
            delta_bias,
            online_z,
        )
        bb, ll, dd = out.shape
        o_prenorm = rearrange(out, 'b d l -> (b l) d')
        if norm_weight is not None:
            z = rearrange(z, 'b d l -> (b l) d')
            out, orsigma = fast_layer_norm.ln_fwd(o_prenorm, z, norm_weight, 1e-6)
        else:
            out = rest[0]
            out = rearrange(out, "b d l -> (b l) d")
        y = rearrange(F.linear(out, out_proj_weight), '(b l) d -> b l d', b=bb)
        if norm_weight is None:
            orsigma = None

        ctx.save_for_backward(xz, conv1d_out, conv1d_weight, conv1d_bias,
                              x_dbl[:, :R].clone(), x_proj_weight, delta_proj_weight, out_proj_weight,
                              Q, K, D, delta, delta_bias, scan_intermediates,
                              norm_weight, o_prenorm, out, orsigma)
        ctx.x_dbl_size = x_dbl.size()
        ctx.has_norm = norm_weight is not None
        return y

    @staticmethod
    # @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."

        (
            xz, conv1d_out, conv1d_weight, conv1d_bias,
            x_dbl_r, x_proj_weight, delta_proj_weight, out_proj_weight,
            Q, K, D, delta, delta_bias, scan_intermediates,
            norm_weight, o_prenorm, oz, orsigma
        ) = ctx.saved_tensors

        L = xz.shape[-1]
        R = delta_proj_weight.shape[1]
        DD = (x_proj_weight.shape[0] - R) // 2
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_proj_weight = torch.einsum("eB,Bd->ed", dout, oz)
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)

        bb, dd, ll = z.shape
        do = rearrange(dout_y.contiguous(), "b d l -> (b l) d")
        if ctx.has_norm:
            z = rearrange(z, 'b d l -> (b l) d').contiguous()
            do, dz_ln_out, do_rms_weight, _, _, _ = fast_layer_norm.ln_bwd(do.contiguous(), o_prenorm, z, orsigma, norm_weight)
            dz_ln_out = rearrange(dz_ln_out, '(b l) d -> b d l', b=bb).contiguous()
            online_z = None
            dz_online = None
        else:
            do_rms_weight = None
            online_z = z
            dz_online = dz

        import online_selective_scan_cuda
        du, dQ, dK, ddelta, dD, dz_online_out, ddelta_bias, dTK = online_selective_scan_cuda.online_bwd(
            conv1d_out,
            Q,
            K,
            delta,
            D,
            delta_bias,
            online_z, # z
            rearrange(do, '(b l) d -> b d l', b=bb).contiguous(),
            scan_intermediates,
            rearrange(o_prenorm, '(b l) d -> b d l', b=bb).contiguous(),
            dz_online
        )
        if not ctx.has_norm:
            dz = dz_online
        dx_dbl = torch.empty(ctx.x_dbl_size, dtype=x_dbl_r.dtype, device=x_dbl_r.device)

        dQ = rearrange(dQ, "b 1 dstate l -> (b l) dstate")#.contiguous()
        dx_dbl[:, -DD:].copy_(dQ)  # (bl d)
        dQ = None

        dK = get_dk(dTK, dK, K)

        dK = rearrange(dK, "b 1 dstate l -> (b l) dstate")#.contiguous()
        dx_dbl[:, R:R+DD].copy_(dK) # (bl d)
        dK = None

        ddelta = rearrange(ddelta, "b d l -> d (b l)")

        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl_r)
        dx_dbl[:, :R].copy_(torch.einsum("dB,dr->Br", ddelta, delta_proj_weight))
        dconv1d_out = rearrange(du, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        _, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        if norm_weight is not None:
            dz.copy_(dz_ln_out)
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        # dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")

        return (dxz, dconv1d_weight.unsqueeze(1), dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                do_rms_weight,
                dout_proj_weight,
                dD.to(D.dtype) if D is not None else None,
                ddelta_bias)


def longhorn_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight, out_proj_weight,
    D=None, delta_bias=None,
):
    return LonghornInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                                      norm_weight, out_proj_weight, D, delta_bias)


def rms_norm_ref(x, weight, eps=1e-6):
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight).to(x.dtype)
    return out


def longhorn_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight, out_proj_weight,
    D=None, delta_bias=None,
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    has_norm_weight = norm_weight is not None
    L = xz.shape[-1]
    R = delta_proj_weight.shape[1]
    DD = (x_proj_weight.shape[0] - R) // 2

    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu").contiguous()
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :R].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    K = x_dbl[:, R:R+DD]  # (bl d)
    K = rearrange(K, "(b l) dstate -> b dstate l", l=L).contiguous()

    Q = x_dbl[:, -DD:]  # (bl d)
    Q = rearrange(Q, "(b l) dstate -> b dstate l", l=L).contiguous()

    if has_norm_weight:
        online_z = None
    else:
        online_z = z

    y = selective_scan_online7_fn(x, Q.to(x), K.to(x), delta.to(x),
                                  D=D,
                                  t_bias=delta_bias,
                                  z=online_z, return_last_state=False)
    y = rearrange(y, "b d l -> b l d")
    if has_norm_weight:
        y = rms_norm_ref(y, norm_weight).to(y) * F.silu(rearrange(z, 'b d l -> b l d')).to(z)
    return F.linear(y, out_proj_weight)


###############################################################################
#
# Bidirectional Longhorn Inner Function
#
###############################################################################


def bi_longhorn_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight_forward, norm_weight_backward, out_proj_weight, D=None, delta_bias=None,
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."

    L = xz.shape[-1]
    R = delta_proj_weight.shape[1]
    DD = (x_proj_weight.shape[0] - R) // 2

    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu").contiguous()
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :R].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    K = x_dbl[:, R:R+DD]  # (bl d)
    K = rearrange(K, "(b l) dstate -> b dstate l", l=L).contiguous()

    Q = x_dbl[:, -DD:]  # (bl d)
    Q = rearrange(Q, "(b l) dstate -> b dstate l", l=L).contiguous()

    y = selective_scan_online7_fn(x, Q.to(x), K.to(x), delta.to(x),
                                  D=D,
                                  t_bias=delta_bias,
                                  z=None, return_last_state=False)
    y_b = selective_scan_online7_fn(x.flip([-1]), Q.to(x).flip([-1]), K.to(x).flip([-1]), delta.to(x).flip([-1]),
                                    D=D,
                                    t_bias=delta_bias,
                                    z=None, return_last_state=False)
    y_b = y_b.flip([-1])
    y = rms_norm_ref(rearrange(y, 'b d l -> b l d'), norm_weight_forward).to(y) + rms_norm_ref(rearrange(y_b, 'b d l -> b l d'), norm_weight_backward).to(y_b)
    y = y * F.silu(rearrange(z, 'b d l -> b l d')).to(z)
    return F.linear(y, out_proj_weight)
