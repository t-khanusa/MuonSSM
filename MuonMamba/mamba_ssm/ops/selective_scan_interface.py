# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd
from mamba_ssm.ops.triton.newton_schulz import newton_schulz_triton_fwd, fast_newtonschulz_batched

import selective_scan_cuda


@torch.compile
def zeropower_via_newtonschulz5(G, steps=1):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    G has shape [..., D, N] where the last two dimensions form the matrix to orthogonalize.
    
    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing
    the slope at zero even beyond the point where the iteration no longer converges all the way to
    one everywhere on the interval. This iteration therefore does not produce UV^T but rather
    something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out
    not to hurt model performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def zeropower_via_newtonschulz5_differentiable(G, steps=1):
    """
    Differentiable version of Newton-Schulz iteration.
    Preserves gradient flow by staying in float32 and using operations that maintain grad_fn.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.float()  # Stay in float32 for gradient flow
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Ensure spectral norm is at most 1 (differentiable)
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm + 1e-7)
    
    # Perform the NS iterations (all differentiable operations)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class NewtonSchulzSTE(torch.autograd.Function):
    """
    Straight-through estimator wrapper for Newton-Schulz orthogonalization.
    Forward: applies Newton-Schulz orthogonalization
    Backward: passes gradients through unchanged (identity)
    """
    @staticmethod
    def forward(ctx, x, steps=1):
        return zeropower_via_newtonschulz5(x, steps=steps).to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Identity gradient (STE), None for steps


def newton_schulz_ste(x, steps=1):
    """Apply Newton-Schulz orthogonalization with straight-through estimator."""
    return NewtonSchulzSTE.apply(x, steps)


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, beta=None, alpha=None):
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
        
        # Set default values for momentum parameters
        if beta is None:
            beta = 0.0  # No momentum by default
        if alpha is None:
            alpha = 1.0
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, 
                                                 float(beta), float(alpha))
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.beta = beta
        ctx.alpha = alpha
        
        # Extract states from x tensor
        # x has shape (batch, dim, n_chunks, dstate * 4) of floats
        # This represents (batch, dim, n_chunks, dstate * 2) of float2 values
        # Reshape to separate float2 components (a, b)
        batch, dim, n_chunks, _ = x.shape
        dstate = A.shape[-1]
        x_reshaped = x.view(batch, dim, n_chunks, dstate * 2, 2)  # (batch, dim, n_chunks, dstate*2, 2)
        
        # Take the 'b' component (index 1) which contains the state values
        states = x_reshaped[:, :, -1, :, 1]  # (batch, dim, dstate * 2)
        
        # Even indices: velocity states, Odd indices: hidden states
        last_velocity = states[:, :, 0::2]  # (batch, dim, dstate)
        last_state = states[:, :, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state, last_velocity)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state, last_velocity)

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
            False,  # option to recompute out_z, not used here
            float(ctx.beta), float(ctx.alpha)
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None,  # dbeta (not computed, beta is fixed)
                None)  # dalpha (not computed, alpha is fixed)


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False, beta=None, alpha=None):
    """if return_last_state is True, returns (out, last_state, last_velocity)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    beta: momentum decay parameter (scalar)
    alpha: momentum scale parameter (scalar)
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, beta, alpha)


class SelectiveScanOrthFn(torch.autograd.Function):
    """
    Selective scan with Newton-Schulz orthogonalization of the velocity input.
    The deltaB_u = alpha * delta * B * u is computed in Python, orthogonalized via NS,
    and then passed to CUDA for the velocity scan.
    
    Uses proper autograd through Newton-Schulz (not STE) for mathematically correct gradients.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, beta=None, alpha=None, ns_steps=1, ns_mode='compile'):
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
        
        squeeze_B = False
        squeeze_C = False
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            squeeze_C = True
        
        # Set default values for momentum parameters
        if beta is None:
            beta = 0.0
        if alpha is None:
            alpha = 1.0
        
        batch, dim, seqlen = u.shape
        dstate = A.shape[-1]
        
        # Apply delta_softplus if needed (for deltaB_u computation)
        delta_for_Bu = delta.float()
        if delta_bias is not None:
            delta_for_Bu = delta_for_Bu + delta_bias[..., None].float()
        if delta_softplus:
            delta_for_Bu = F.softplus(delta_for_Bu)
        
        # Compute deltaB_u = alpha * delta * B * u  ->  shape [B, D, L, N]
        # B has shape (batch, 1, dstate, seqlen) for variable B
        # Expand B to match dim
        B_expanded = repeat(B, "b g n l -> b (g d) l n", d=dim // B.shape[1])  # (B, D, L, N)
        deltaB_u = alpha * delta_for_Bu.unsqueeze(-1) * B_expanded * u.unsqueeze(-1)  # (B, D, L, N)
        
        # Apply Newton-Schulz orthogonalization (NO STE - use differentiable version)
        # NS operates on shape [B, L, D, N] to orthogonalize (D, N) matrices
        deltaB_u_for_ns = rearrange(deltaB_u, "b d l n -> b l d n")
        
        # Select NS implementation based on mode
        if ns_mode == 'triton':
            deltaB_u_orth_ns = newton_schulz_triton_fwd(deltaB_u_for_ns, steps=ns_steps)
        elif ns_mode == 'flash_muon':
            deltaB_u_orth_ns = fast_newtonschulz_batched(deltaB_u_for_ns, steps=ns_steps)
        else:  # 'compile' (default)
            deltaB_u_orth_ns = zeropower_via_newtonschulz5(deltaB_u_for_ns, steps=ns_steps)
        deltaB_u_orth_ns = deltaB_u_orth_ns.to(deltaB_u.dtype)
        # Rearrange back to [B, D, L, N] for CUDA
        deltaB_u_orth = rearrange(deltaB_u_orth_ns, "b l d n -> b d l n").contiguous()
        
        # Call CUDA with orthogonalized input
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, 
                                                 float(beta), float(alpha), deltaB_u_orth)
        
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.squeeze_B = squeeze_B
        ctx.squeeze_C = squeeze_C
        ctx.ns_steps = ns_steps
        ctx.ns_mode = ns_mode
        
        # Extract states from x tensor
        batch, dim, n_chunks, _ = x.shape
        x_reshaped = x.view(batch, dim, n_chunks, dstate * 2, 2)
        states = x_reshaped[:, :, -1, :, 1]
        last_velocity = states[:, :, 0::2]
        last_state = states[:, :, 1::2]
        
        # Save tensors needed for backward
        # MEMORY OPTIMIZATION: Don't save deltaB_u_orth (saves ~576 MB per layer!)
        # Instead, recompute it in backward pass from saved u, delta, A, B, C, D, delta_bias
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state, last_velocity)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state, last_velocity)

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
        
        # MEMORY OPTIMIZATION: Recompute deltaB_u_orth instead of loading from saved tensors
        batch, dim, seqlen = u.shape
        dstate = A.shape[-1]
        ngroups = B.shape[1]
        
        # Recompute delta after softplus (same as forward)
        delta_for_Bu = delta.float()
        if delta_bias is not None:
            delta_for_Bu = delta_for_Bu + delta_bias[..., None].float()
        if ctx.delta_softplus:
            delta_for_Bu = F.softplus(delta_for_Bu)
        
        # Recompute deltaB_u = alpha * delta * B * u  ->  shape [B, D, L, N]
        B_expanded = repeat(B, "b g n l -> b (g d) l n", d=dim // ngroups)  # (B, D, L, N)
        deltaB_u = ctx.alpha * delta_for_Bu.unsqueeze(-1) * B_expanded * u.unsqueeze(-1).float()  # (B, D, L, N)
        
        # Apply Newton-Schulz orthogonalization (same as forward, but without grad tracking for CUDA call)
        deltaB_u_for_ns = rearrange(deltaB_u, "b d l n -> b l d n")
        
        # Select NS implementation based on mode (same as forward)
        with torch.no_grad():
            if ctx.ns_mode == 'triton':
                deltaB_u_orth_ns = newton_schulz_triton_fwd(deltaB_u_for_ns, steps=ctx.ns_steps)
            elif ctx.ns_mode == 'flash_muon':
                deltaB_u_orth_ns = fast_newtonschulz_batched(deltaB_u_for_ns, steps=ctx.ns_steps)
            else:  # 'compile' (default)
                deltaB_u_orth_ns = zeropower_via_newtonschulz5(deltaB_u_for_ns, steps=ctx.ns_steps)
        
        # Rearrange back to [B, D, L, N] for CUDA (convert to float32 for CUDA kernel)
        deltaB_u_orth = rearrange(deltaB_u_orth_ns, "b l d n -> b d l n").contiguous().float()
        
        # Call CUDA backward with recomputed orthogonalized input
        results = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False,  # option to recompute out_z
            float(ctx.beta), float(ctx.alpha), deltaB_u_orth
        )
        
        # Free deltaB_u_orth immediately after CUDA backward call
        del deltaB_u_orth, deltaB_u_for_ns, deltaB_u, B_expanded, delta_for_Bu
        
        # Results: du, ddelta, dA, dB, dC, dD, ddelta_bias, [dz], [out_z], [d_deltaB_u_orth]
        du, ddelta, dA, dB, dC, dD, ddelta_bias = results[:7]
        
        rest_idx = 7
        dz = None
        if ctx.has_z:
            dz = results[rest_idx]
            rest_idx += 1
        
        # d_deltaB_u_orth is the last tensor in results - gradient w.r.t. orthogonalized input
        d_deltaB_u_orth = results[-1]  # Shape: [B, D, L, N]
        
        # Rearrange to [B, L, D, N] to match NS input shape
        d_deltaB_u_orth_ns = rearrange(d_deltaB_u_orth, "b d l n -> b l d n")
        
        # ================================================================
        # PROPER AUTOGRAD THROUGH NEWTON-SCHULZ (not STE)
        # Recompute deltaB_u_for_ns (to save memory - not saved in forward)
        # then run NS forward with requires_grad to build computation graph
        # ================================================================
        
        batch, dim, seqlen = u.shape
        dstate = A.shape[-1]
        ngroups = B.shape[1]
        
        # Recompute delta after softplus (same as forward)
        delta_for_Bu = delta.float()
        if delta_bias is not None:
            delta_for_Bu = delta_for_Bu + delta_bias[..., None].float()
        if ctx.delta_softplus:
            delta_for_Bu = F.softplus(delta_for_Bu)
        
        # Recompute deltaB_u = alpha * delta * B * u  (same as forward)
        B_expanded = repeat(B, "b g n l -> b (g d) l n", d=dim // ngroups)
        deltaB_u = ctx.alpha * delta_for_Bu.unsqueeze(-1) * B_expanded * u.unsqueeze(-1).float()
        
        # Rearrange for NS input [B, L, D, N]
        deltaB_u_for_ns = rearrange(deltaB_u, "b d l n -> b l d n")
        
        # Create tensor with requires_grad for autograd
        deltaB_u_for_ns_grad = deltaB_u_for_ns.detach().clone().requires_grad_(True)
        
        # Re-run Newton-Schulz forward to build computation graph (use differentiable version)
        with torch.enable_grad():
            deltaB_u_orth_recomputed = zeropower_via_newtonschulz5_differentiable(deltaB_u_for_ns_grad, steps=ctx.ns_steps)
        
        # Make sure grad_outputs is contiguous and correct dtype
        grad_outputs = d_deltaB_u_orth_ns.contiguous().float()
        
        # Backprop through NS using torch.autograd.grad
        d_deltaB_u = torch.autograd.grad(
            outputs=deltaB_u_orth_recomputed,
            inputs=deltaB_u_for_ns_grad,
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False
        )[0]
        
        # d_deltaB_u has shape [B, L, D, N], rearrange back to [B, D, L, N]
        d_deltaB_u = rearrange(d_deltaB_u, "b l d n -> b d l n")
        
        # Now compute gradients for delta, B, u from d_deltaB_u
        # deltaB_u = alpha * delta * B * u
        # d_delta = alpha * d_deltaB_u * B * u  (summed over dstate)
        # d_B = alpha * d_deltaB_u * delta * u
        # d_u = alpha * d_deltaB_u * delta * B  (summed over dstate)
        # Note: B_expanded and delta_for_Bu already computed above for NS recomputation
        
        # d_delta contribution from velocity path (through NS autograd)
        d_delta_from_orth = ctx.alpha * (d_deltaB_u * B_expanded * u.unsqueeze(-1).float()).sum(dim=-1)
        
        # d_u contribution from velocity path
        d_u_from_orth = ctx.alpha * (d_deltaB_u * delta_for_Bu.unsqueeze(-1) * B_expanded).sum(dim=-1)
        
        # d_B contribution from velocity path
        # d_B_from_orth has shape (B, D, L, N), need to reduce to (B, G, N, L)
        d_B_from_orth = ctx.alpha * (d_deltaB_u * delta_for_Bu.unsqueeze(-1) * u.unsqueeze(-1).float())
        # Reshape to (B, G, D//G, L, N) and sum over dim groups
        d_B_from_orth = rearrange(d_B_from_orth, "b (g d) l n -> b g d l n", g=ngroups)
        d_B_from_orth = d_B_from_orth.sum(dim=2)  # (B, G, L, N)
        d_B_from_orth = rearrange(d_B_from_orth, "b g l n -> b g n l")  # (B, G, N, L)
        
        # Chain rule through softplus if needed
        if ctx.delta_softplus:
            delta_val = delta.float()
            if delta_bias is not None:
                delta_val = delta_val + delta_bias[..., None].float()
            softplus_grad = torch.where(delta_val <= 20.0, torch.sigmoid(delta_val), torch.ones_like(delta_val))
            d_delta_from_orth = d_delta_from_orth * softplus_grad
        
        # Combine with CUDA-computed gradients (from exp term)
        # Note: When using orth input, CUDA's dB is incorrect (B wasn't used in CUDA)
        du_combined = du + d_u_from_orth.to(du.dtype)
        ddelta_combined = ddelta + d_delta_from_orth.to(ddelta.dtype)
        # Only use Python-computed dB (CUDA doesn't know about B when use_orth_input=true)
        dB_combined = d_B_from_orth.to(dB.dtype)
        
        dB_combined = dB_combined.squeeze(1) if ctx.squeeze_B else dB_combined
        dC = dC.squeeze(1) if ctx.squeeze_C else dC
        
        return (du_combined, ddelta_combined, dA, dB_combined, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,  # delta_softplus
                None,  # return_last_state
                None,  # beta
                None,  # alpha
                None,  # ns_steps
                None)  # ns_mode


def selective_scan_fn_orth(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                          return_last_state=False, beta=None, alpha=None, ns_steps=1, ns_mode='compile'):
    """
    Selective scan with Newton-Schulz orthogonalization of the velocity input.
    
    This version computes deltaB_u = alpha * delta * B * u in Python,
    applies Newton-Schulz orthogonalization, and passes the orthogonalized tensor to CUDA for the velocity scan.
    
    Args:
        u: Input tensor (B, D, L)
        delta: Delta tensor (B, D, L)
        A: State transition matrix (D, N)
        B: Input matrix (B, G, N, L) or (D, N)
        C: Output matrix (B, G, N, L) or (D, N)
        D: Skip connection (D,)
        z: Gate tensor (B, D, L)
        delta_bias: Bias for delta (D,)
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return last states
        beta: Momentum decay parameter
        alpha: Momentum scale parameter
        ns_steps: Number of Newton-Schulz iterations (default 1)
        ns_mode: NS implementation mode: 'compile' (torch.compile), 'triton', or 'flash_muon'
    
    Returns:
        out: Output tensor (B, D, L) or (out, last_state, last_velocity) if return_last_state
    """
    return SelectiveScanOrthFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus,
                                     return_last_state, beta, alpha, ns_steps, ns_mode)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, beta=0.0, alpha=1.0):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    beta: momentum decay (scalar)
    alpha: momentum scale (scalar)

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    last_velocity (optional): r(B D dstate) or c(B D dstate)
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
    v = A.new_zeros((batch, dim, dstate))  # velocity state
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
    last_velocity = None
    for i in range(u.shape[2]):
        # Momentum: v_t = beta * v_{t-1} + alpha * B_t * x_t
        v = beta * v + alpha * deltaB_u[:, :, i]
        # Hidden state: h_t = A_t * h_{t-1} + v_t
        x = deltaA[:, :, i] * x + v
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
            last_velocity = v
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state, last_velocity)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
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
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
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
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
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
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)


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
