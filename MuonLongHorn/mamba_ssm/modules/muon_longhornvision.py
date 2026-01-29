"""
MuonLonghorn Vision Mixer - Longhorn with momentum and Newton-Schulz for vision tasks.

This module provides MuonLonghornVisionMixer, which replaces the Mamba SSM in MambaVisionMixer
with MuonLonghorn's SSM (momentum + Newton-Schulz orthogonalization) while keeping the 
MambaVision-compatible architecture (split x/z branches, dual convolutions, concatenation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
    selective_scan_online7_fn,
    selective_scan_online_orth_fn,
)



class MuonLonghornVisionMixer(nn.Module):
    """
    MuonLonghorn Vision Mixer: Longhorn SSM with momentum and Newton-Schulz for vision tasks.
    
    This module keeps the MambaVisionMixer architecture (split x/z, dual conv1d, concatenation)
    but replaces the traditional Mamba SSM core with MuonLonghorn's SSM:
    - Uses Q, K projections instead of B, C (attention-like SSM)
    - Adds momentum velocity state: v_t = β * v_{t-1} + α * input
    - Optionally applies Newton-Schulz orthogonalization for stability
    
    Architecture:
        Input: (B, L, D)
        └── in_proj → (B, L, d_inner)
            └── split into x, z each (B, L, d_inner//2)
                ├── x: conv1d_x → SiLU → MuonLonghorn SSM → y
                └── z: conv1d_z → SiLU (gating branch)
            └── concat(y, z) → out_proj → (B, L, D)
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        dt_rank: Rank of delta projection, 'auto' = ceil(d_model / 16)
        dt_min: Minimum delta value for initialization
        dt_max: Maximum delta value for initialization
        dt_init: Delta initialization mode ('random' or 'constant')
        dt_scale: Delta scale factor
        dt_init_floor: Floor for delta initialization
        conv_bias: Whether to use bias in convolution
        bias: Whether to use bias in linear layers
        use_fast_path: Whether to use fused kernel (placeholder)
        layer_idx: Layer index for caching
        
        # MuonLonghorn-specific parameters:
        beta: Velocity decay factor (momentum), β ∈ [0, 1] (default: 0.9)
              - β = 0: No momentum (equivalent to vanilla Longhorn)
              - β = 0.9: High momentum, smooth state updates
        alpha: Velocity scale factor, α > 0 (default: 1.0)
        use_newton_schulz: Whether to apply Newton-Schulz orthogonalization (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 1)
        ns_mode: Newton-Schulz mode - 'compile' or 'triton' (default: 'compile')
    
    Example:
        >>> mixer = MuonLonghornVisionMixer(
        ...     d_model=256,
        ...     d_state=16,
        ...     beta=0.9,
        ...     alpha=1.0,
        ...     use_newton_schulz=True,
        ...     device='cuda',
        ... )
        >>> x = torch.randn(2, 196, 256, device='cuda')  # (batch, seq_len, dim)
        >>> y = mixer(x)  # (2, 196, 256)
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        # MuonLonghorn-specific parameters
        # Note: Start with conservative defaults for stability
        beta=0.0,  # No momentum by default (0.9 for full momentum)
        alpha=1.0,
        use_newton_schulz=False,  # Disable NS by default for stability
        ns_steps=1,
        ns_mode='compile',
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Core dimensions
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_ssm = self.d_inner // 2  # SSM operates on half the inner dimension
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # MuonLonghorn-specific: momentum and Newton-Schulz parameters
        self.beta = beta
        self.alpha = alpha
        self.use_newton_schulz = use_newton_schulz
        self.ns_steps = ns_steps
        self.ns_mode = ns_mode
        
        # Input projection: d_model -> d_inner (split into x and z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # MuonLonghorn SSM: x_proj outputs dt_rank + 2*d_state (for dt, K, Q)
        # Note: Longhorn uses K, Q instead of B, C
        self.x_proj = nn.Linear(
            self.d_ssm, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        
        # Delta (time step) projection: dt_rank -> d_ssm
        self.dt_proj = nn.Linear(self.dt_rank, self.d_ssm, bias=True, **factory_kwargs)
        
        # Initialize dt_proj weights
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init must be 'constant' or 'random', got {dt_init}")
        
        # Initialize dt_proj bias (similar to Longhorn's initialization)
        dt = torch.exp(
            torch.rand(self.d_ssm, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # D "skip" parameter (Longhorn-style)
        self.D = nn.Parameter(torch.ones(self.d_ssm, device=device))
        self.D._no_weight_decay = True
        
        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Dual convolutions (MambaVision-style: separate conv for x and z branches)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_ssm,
            out_channels=self.d_ssm,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_ssm,
            padding=d_conv - 1,  # causal padding
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_ssm,
            out_channels=self.d_ssm,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_ssm,
            padding=d_conv - 1,  # causal padding
            **factory_kwargs,
        )

    def forward(self, hidden_states, inference_params=None):
        """
        Forward pass through MuonLonghornVisionMixer.
        
        Args:
            hidden_states: (B, L, D) input tensor
            inference_params: Optional inference parameters (for generation)
        
        Returns:
            (B, L, D) output tensor
        """
        # Store input dtype for mixed precision compatibility
        input_dtype = hidden_states.dtype
        batch, seqlen, _ = hidden_states.shape
        
        # Input projection and split into x (SSM branch) and z (gating branch)
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l").contiguous()
        x, z = xz.chunk(2, dim=1)  # Each: (B, d_ssm, L)
        
        # Apply convolutions with SiLU activation
        # Using 'same' padding to maintain sequence length
        x = F.silu(F.conv1d(
            input=x.contiguous(), 
            weight=self.conv1d_x.weight, 
            bias=self.conv1d_x.bias, 
            padding='same', 
            groups=self.d_ssm
        ))
        z = F.silu(F.conv1d(
            input=z.contiguous(), 
            weight=self.conv1d_z.weight, 
            bias=self.conv1d_z.bias, 
            padding='same', 
            groups=self.d_ssm
        ))
        
        # For SSM operations, we need float32 for numerical stability
        # CUDA kernel requires all inputs to have the same dtype
        x_float = x.float().contiguous()
        
        # Project x to get dt, K, Q (Longhorn-style: K and Q instead of B and C)
        x_dbl = self.x_proj(rearrange(x_float, "b d l -> (b l) d"))  # (B*L, dt_rank + 2*d_state)
        dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt through dt_proj
        dt = self.dt_proj.weight @ dt.t()  # (d_ssm, B*L)
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen).contiguous()
        
        # Rearrange K and Q for SSM
        k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # CRITICAL: Ensure all tensors have same dtype (float32) for CUDA kernel
        # The CUDA kernel requires: u.dtype == Q.dtype == K.dtype == T.dtype
        dt = dt.float().contiguous()
        k = k.float().contiguous()
        q = q.float().contiguous()
        
        # MuonLonghorn SSM: select appropriate scan function based on Newton-Schulz setting
        # SSM operations in float32 for numerical stability
        if self.use_newton_schulz:
            y = selective_scan_online_orth_fn(
                x_float,
                q,
                k,
                dt,
                D=self.D.float(),
                t_bias=self.dt_proj.bias.float(),
                z=None,  # z is concatenated later (MambaVision-style)
                return_last_state=False,
                beta=self.beta,
                alpha=self.alpha,
                ns_steps=self.ns_steps,
                ns_mode=self.ns_mode,
            )
        else:
            y = selective_scan_online7_fn(
                x_float,
                q,
                k,
                dt,
                D=self.D.float(),
                t_bias=self.dt_proj.bias.float(),
                z=None,  # z is concatenated later (MambaVision-style)
                return_last_state=False,
                beta=self.beta,
                alpha=self.alpha,
            )
        
        # Convert back to input dtype for mixed precision compatibility
        y = y.to(dtype=input_dtype).contiguous()
        z = z.to(dtype=input_dtype).contiguous()
        
        # MambaVision-style: concatenate SSM output with gating branch
        y = torch.cat([y, z], dim=1)  # (B, d_inner, L)
        y = rearrange(y, "b d l -> b l d").contiguous()
        
        # Output projection
        out = self.out_proj(y)
        
        # Ensure output is contiguous and has correct dtype for downstream operations
        return out.contiguous()
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate inference cache for incremental generation.
        
        Returns:
            (conv_state, ssm_state, velocity_state) tuple
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d_x.weight.dtype if dtype is None else dtype
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        
        conv_state_x = torch.zeros(
            batch_size, self.d_ssm, self.d_conv, device=device, dtype=conv_dtype
        )
        conv_state_z = torch.zeros(
            batch_size, self.d_ssm, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_state = torch.zeros(
            batch_size, self.d_ssm, self.d_state, device=device, dtype=ssm_dtype
        )
        # MuonLonghorn: velocity state for momentum
        velocity_state = torch.zeros(
            batch_size, self.d_ssm, self.d_state, device=device, dtype=ssm_dtype
        )
        
        return (conv_state_x, conv_state_z), ssm_state, velocity_state


# Alias for easier import
LonghornVisionMixer = MuonLonghornVisionMixer
