import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules.activations import ACT2FN
from .gated_delta_rule_ops import chunk_gated_delta_rule
try:
    from fla.modules.l2norm import l2_norm as l2_norm_fn
except ImportError:
    from fla.modules.l2norm import l2_norm_fn



class MuonGatedDeltaNetVisionMixer(nn.Module):
    """
    MuonGatedDeltaNet Vision Mixer: Gated Delta Net SSM with momentum for vision tasks.
    
    This module keeps the MambaVisionMixer architecture (split x/z, dual conv1d, concatenation)
    but replaces the traditional Mamba SSM core with Gated Delta Net's SSM:
    - Uses Q, K, V projections with multi-head attention-like structure
    - Applies gated delta rule: S_t = S_{t-1} * exp(gk_t) + beta_t * k_t * v_t^T
    - Supports different QK normalization: 'l2', 'longhorn', 'softmax'
    - Uses Mamba-style gating for the gate key (gk)
    - Adds momentum parameters for potential future momentum-enhanced delta rule
    
    Architecture:
        Input: (B, L, D)
        └── in_proj → (B, L, d_inner)
            └── split into x, z each (B, L, d_inner//2)
                ├── x: conv1d_x → SiLU → Gated Delta Net SSM → y
                └── z: conv1d_z → SiLU (gating branch)
            └── concat(y, z) → out_proj → (B, L, D)
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (used for head_qk_dim calculation, default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        expand_k: Expansion factor for key dimension (default: 0.75)
        expand_v: Expansion factor for value dimension (default: 1.5)
        num_heads: Number of attention heads (default: 4)
        qk_norm: QK normalization mode - 'l2', 'longhorn', or 'softmax' (default: 'l2')
        gate_fn: Gate activation function (default: 'swish')
        gate_logit_normalizer: Normalizer for gate logits (default: 16)
        use_mamba_gate: Whether to use Mamba-style gating (default: True)
        conv_bias: Whether to use bias in convolution
        bias: Whether to use bias in linear layers
        layer_idx: Layer index for caching
        
        # Momentum parameters (for future momentum-enhanced delta rule):
        momentum_beta: Velocity decay factor (momentum), β ∈ [0, 1] (default: 0.9)
        momentum_alpha: Velocity scale factor, α > 0 (default: 1.0)
    
    Example:
        >>> mixer = MuonGatedDeltaNetVisionMixer(
        ...     d_model=256,
        ...     num_heads=4,
        ...     qk_norm='l2',
        ...     use_mamba_gate=True,
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
        expand_k=0.75,
        expand_v=1.5,
        num_heads=4,
        qk_norm='l2',
        gate_fn='swish',
        gate_logit_normalizer=16,
        use_mamba_gate=True,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        # Momentum parameters
        momentum_beta=0.9,
        momentum_alpha=1.0,
        # Additional Gated Delta Net parameters
        elementwise_affine=True,
        norm_eps=1e-5,
        fuse_norm=True,
        use_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Validate qk_norm
        assert qk_norm in ['l2', 'longhorn', 'softmax'], f"qk_norm must be 'l2', 'longhorn', or 'softmax', got {qk_norm}"
        
        # Core dimensions
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_ssm = self.d_inner // 2  # SSM operates on half the inner dimension
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # Gated Delta Net specific dimensions
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.gate_logit_normalizer = gate_logit_normalizer
        self.use_mamba_gate = use_mamba_gate
        self.use_residual = use_residual
        
        # Key and value dimensions based on d_ssm (the SSM branch dimension)
        self.key_dim = int(self.d_ssm * expand_k)
        self.value_dim = int(self.d_ssm * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        # Momentum parameters (stored as buffers)
        self.register_buffer("momentum_beta", torch.tensor(momentum_beta, dtype=torch.float32, device=device))
        self.register_buffer("momentum_alpha", torch.tensor(momentum_alpha, dtype=torch.float32, device=device))
        
        # Input projection: d_model -> d_inner (split into x and z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # Gated Delta Net projections (Q, K, V, G from x branch)
        self.q_proj = nn.Linear(self.d_ssm, self.key_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_ssm, self.key_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_ssm, self.value_dim, bias=False, **factory_kwargs)
        self.g_proj = nn.Linear(self.d_ssm, self.value_dim, bias=False, **factory_kwargs)
        
        # Gate key projection (Mamba-style or logsigmoid)
        self.gk_proj = nn.Linear(self.d_ssm, self.num_heads, bias=not use_mamba_gate, **factory_kwargs)
        
        # Beta projection for delta rule
        self.b_proj = nn.Linear(self.d_ssm, self.num_heads, bias=True, **factory_kwargs)
        
        # Mamba-style gating parameters
        if use_mamba_gate:
            A = torch.empty(self.num_heads, dtype=torch.float32, device=device).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            
            self.D = nn.Parameter(torch.ones(self.num_heads, device=device))
            self.D._no_weight_decay = True
            
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads, device=device) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True
        
        # Residual D parameter (if use_residual)
        if use_residual and not use_mamba_gate:
            self.D = nn.Parameter(torch.ones(self.num_heads, device=device))
            self.D._no_weight_decay = True
        
        # Output normalization and gating
        if gate_fn == 'swish' and fuse_norm:
            from fla.modules import FusedRMSNormSwishGate
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            from fla.modules import RMSNorm
            from fla.modules.activations import ACT2FN
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]
        
        # Output projection from value_dim back to d_ssm
        self.ssm_out_proj = nn.Linear(self.value_dim, self.d_ssm, bias=False, **factory_kwargs)
        
        # Final output projection: d_inner -> d_model
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
        Forward pass through MuonGatedDeltaNetVisionMixer.
        
        Args:
            hidden_states: (B, L, D) input tensor
        
        Returns:
            (B, L, D) output tensor
        """
        from .gated_delta_rule_ops import chunk_gated_delta_rule
        try:
            from fla.modules.l2norm import l2_norm as l2_norm_fn
        except:
            from fla.modules.l2norm import l2_norm_fn
        
        batch, seqlen, _ = hidden_states.shape
        
        # Input projection and split into x (SSM branch) and z (gating branch)
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  # Each: (B, d_ssm, L)
        
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
        
        # Rearrange x for Gated Delta Net projections: (B, d_ssm, L) -> (B, L, d_ssm)
        x = rearrange(x, "b d l -> b l d")
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, key_dim)
        k = self.k_proj(x)  # (B, L, key_dim)
        v = self.v_proj(x)  # (B, L, value_dim)
        
        # Gate key projection
        gk = self.gk_proj(x).float()  # (B, L, num_heads)
        if self.use_mamba_gate:
            gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        else:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        gk = gk.transpose(1, 2)  # (B, num_heads, L)
        
        # Beta projection for delta rule
        beta = self.b_proj(x).float().sigmoid()  # (B, L, num_heads)
        beta = beta.transpose(1, 2)  # (B, num_heads, L)
        
        # Rearrange Q, K, V for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)  # (B, H, L, head_qk_dim)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)  # (B, H, L, head_qk_dim)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)  # (B, H, L, head_v_dim)
        
        # Apply QK normalization
        if self.qk_norm == 'l2':
            q = l2_norm_fn(q).to(v)
            k = l2_norm_fn(k).to(v)
        elif self.qk_norm == 'softmax':
            q = q.softmax(dim=-1).to(v)
            k = k.softmax(dim=-1).to(v)
        elif self.qk_norm == 'longhorn':
            # Longhorn-style normalization adjusts beta based on k norm
            beta = beta / (1 + beta * (k * k).sum(-1))
        
        # Apply Gated Delta Rule SSM
        o, _ = chunk_gated_delta_rule(
            q, k, v, beta, gk,
            initial_state=None,
            output_final_state=False
        )  # o: (B, H, L, head_v_dim)
        
        # Add residual connection if enabled
        if self.use_residual:
            o = o + self.D[None, :, None, None] * v
        
        # Rearrange for output normalization
        o = rearrange(o, 'b h l d -> b l h d')  # (B, L, H, head_v_dim)
        
        # Apply output gating with normalization
        g = self.g_proj(x)  # (B, L, value_dim)
        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        
        # Project back to d_ssm dimension
        y = self.ssm_out_proj(o)  # (B, L, d_ssm)
        
        # Rearrange y back to (B, d_ssm, L) for concatenation
        y = rearrange(y, "b l d -> b d l")
        
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
            (conv_state, ssm_state) tuple
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d_x.weight.dtype if dtype is None else dtype
        ssm_dtype = self.q_proj.weight.dtype if dtype is None else dtype
        
        conv_state_x = torch.zeros(
            batch_size, self.d_ssm, self.d_conv, device=device, dtype=conv_dtype
        )
        conv_state_z = torch.zeros(
            batch_size, self.d_ssm, self.d_conv, device=device, dtype=conv_dtype
        )
        # SSM state for Gated Delta Net: (B, num_heads, head_qk_dim, head_v_dim)
        ssm_state = torch.zeros(
            batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim, 
            device=device, dtype=ssm_dtype
        )
        
        return (conv_state_x, conv_state_z), ssm_state