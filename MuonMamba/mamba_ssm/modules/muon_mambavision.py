import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
    selective_scan_fn_orth,
)


class MuonMambaVisionMixer(nn.Module):
    """
    MuonMamba Vision Mixer: Traditional Mamba SSM with momentum and Newton-Schulz for vision tasks.
    
    This module keeps the MambaVisionMixer architecture (split x/z, dual conv1d, concatenation)
    but adds MuonMamba's momentum and Newton-Schulz orthogonalization features:
    - Uses traditional A, B, C, D SSM parameters (like original Mamba)
    - Adds momentum velocity state: v_t = β * v_{t-1} + α * B_t * x_t
    - Optionally applies Newton-Schulz orthogonalization for stability
    
    Architecture:
        Input: (B, L, D)
        └── in_proj → (B, L, d_inner)
            └── split into x, z each (B, L, d_inner//2)
                ├── x: conv1d_x → SiLU → MuonMamba SSM (A,B,C,D + momentum) → y
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
        use_fast_path: Whether to use fused kernel (only when beta=0 and not using NS)
        layer_idx: Layer index for caching
        
        # MuonMamba-specific parameters:
        beta: Velocity decay factor (momentum), β ∈ [0, 1] (default: 0.9)
              - β = 0: No momentum (equivalent to vanilla Mamba)
              - β = 0.9: High momentum, smooth state updates
        alpha: Velocity scale factor, α > 0 (default: 1.0)
        use_newton_schulz: Whether to apply Newton-Schulz orthogonalization (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 1)
    
    Example:
        >>> mixer = MuonMambaVisionMixer(
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
        # MuonMamba-specific parameters
        beta=0.9,
        alpha=1.0,
        use_newton_schulz=True,
        ns_steps=1,
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
        
        # MuonMamba-specific: momentum and Newton-Schulz parameters
        self.use_newton_schulz = use_newton_schulz
        self.ns_steps = ns_steps
        
        # Register momentum parameters as buffers (not learnable)
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32, device=device))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32, device=device))
        
        # Input projection: d_model -> d_inner (split into x and z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # SSM projections: x_proj outputs dt_rank + 2*d_state (for dt, B, C)
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
        
        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_ssm, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A parameter (traditional Mamba-style: negative log of decay)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_ssm,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
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
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_ssm,
            out_channels=self.d_ssm,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_ssm,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        Forward pass through MuonMambaVisionMixer.
        
        Args:
            hidden_states: (B, L, D) input tensor
        
        Returns:
            (B, L, D) output tensor
        """
        batch, seqlen, _ = hidden_states.shape
        
        # Input projection and split into x (SSM branch) and z (gating branch)
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  # Each: (B, d_ssm, L)
        
        # Compute A from A_log (negative exponential for stability)
        A = -torch.exp(self.A_log.float())
        
        # Apply convolutions with SiLU activation
        x = F.silu(F.conv1d(
            input=x, 
            weight=self.conv1d_x.weight, 
            bias=self.conv1d_x.bias, 
            padding='same', 
            groups=self.d_ssm
        ))
        z = F.silu(F.conv1d(
            input=z, 
            weight=self.conv1d_z.weight, 
            bias=self.conv1d_z.bias, 
            padding='same', 
            groups=self.d_ssm
        ))
        
        # Project x to get dt, B, C (traditional Mamba-style)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (B*L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt through dt_proj
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        
        # Rearrange B and C for SSM
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # MuonMamba SSM: select appropriate scan function based on Newton-Schulz setting
        if self.use_newton_schulz:
            # MuonMamba with Newton-Schulz orthogonalization
            y = selective_scan_fn_orth(
                x, 
                dt, 
                A, 
                B, 
                C, 
                self.D.float(), 
                z=None,  # z is concatenated later (MambaVision-style)
                delta_bias=self.dt_proj.bias.float(), 
                delta_softplus=True, 
                return_last_state=None,
                beta=self.beta,
                alpha=self.alpha,
                ns_steps=self.ns_steps,
            )
        else:
            # MuonMamba with momentum only (no Newton-Schulz)
            y = selective_scan_fn(
                x, 
                dt, 
                A, 
                B, 
                C, 
                self.D.float(), 
                z=None,  # z is concatenated later (MambaVision-style)
                delta_bias=self.dt_proj.bias.float(), 
                delta_softplus=True, 
                return_last_state=None,
                beta=self.beta,
                alpha=self.alpha,
            )
        
        # MambaVision-style: concatenate SSM output with gating branch
        y = torch.cat([y, z], dim=1)  # (B, d_inner, L)
        y = rearrange(y, "b d l -> b l d")
        
        # Output projection
        out = self.out_proj(y)
        return out
    
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
        # MuonMamba: velocity state for momentum
        velocity_state = torch.zeros(
            batch_size, self.d_ssm, self.d_state, device=device, dtype=ssm_dtype
        )
        
        return (conv_state_x, conv_state_z), ssm_state, velocity_state
