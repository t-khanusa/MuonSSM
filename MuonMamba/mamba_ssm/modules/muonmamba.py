# Copyright (c) 2023, Tri Dao, Albert Gu.
# MuonMamba: Multi-layer Mamba with Momentum and Newton-Schulz Orthogonalization

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm as TritonRMSNorm
except ImportError:
    TritonRMSNorm = None


class RMSNorm(nn.Module):
    """RMSNorm implementation - fallback if Triton version not available"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


@dataclass
class MuonMambaConfig:
    """
    Configuration for MuonMamba: Multi-layer Mamba with Momentum and Newton-Schulz Orthogonalization
    
    Key parameters:
    - momentum_beta: β ∈ [0, 1] - Controls velocity decay (0 = no momentum, 0.9 = high momentum)
    - momentum_alpha: α > 0 - Scales the velocity contribution (typically 0.5-1.5)
    - use_newton_schulz: When True, NS orthogonalization is applied to stabilize momentum
    """
    d_model: int  # D - model dimension
    n_layers: int  # Number of layers
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N - SSM state dimension
    expand_factor: int = 2  # E - expansion factor
    d_conv: int = 4  # Convolution kernel size
    
    # SSM discretization parameters
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    
    # MuonMamba: Momentum + Newton-Schulz parameters
    momentum_beta: float = 0.9  # β - momentum decay factor (0 = no momentum)
    momentum_alpha: float = 1.0  # α - momentum input scaling
    use_newton_schulz: bool = True  # Enable Newton-Schulz orthogonalization
    ns_steps: int = 1  # Number of Newton-Schulz iterations
    
    # Normalization
    rms_norm_eps: float = 1e-5
    
    # Architecture options
    bias: bool = False
    conv_bias: bool = True
    use_fast_path: bool = True  # Use fused CUDA kernels (only when beta=0)
    
    # Initialization parameters
    base_std: float = 0.02  # Base standard deviation for weight initialization
    initializer_range: float = 0.02  # Range for uniform initialization
    rescale_prenorm_residual: bool = True  # Scale output projection by 1/sqrt(2*n_layers)
    
    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        # Validate momentum parameters
        if not (0.0 <= self.momentum_beta <= 1.0):
            raise ValueError(f"momentum_beta must be in [0, 1], got {self.momentum_beta}")
        
        if self.momentum_alpha <= 0:
            raise ValueError(f"momentum_alpha must be positive, got {self.momentum_alpha}")


class MuonMamba(nn.Module):
    """
    Multi-layer MuonMamba model with residual connections
    
    MuonMamba = Mamba + Momentum + Newton-Schulz Orthogonalization
    
    Architecture:
    - Each layer: RMSNorm → Mamba → Residual connection
    - Mamba: Uses momentum-based SSM with optional NS orthogonalization
    
    Uses the Mamba implementation from mamba_simple.py
    """
    def __init__(self, config: MuonMambaConfig):
        super().__init__()
        
        self.config = config
        self.n_layers = config.n_layers
        
        self.layers = nn.ModuleList([
            ResidualBlock(config, layer_idx=i) 
            for i in range(config.n_layers)
        ])
        
        # Final norm (optional, but helps with training stability)
        if TritonRMSNorm is not None:
            self.final_norm = TritonRMSNorm(config.d_model, eps=config.rms_norm_eps)
        else:
            self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections (scale by depth)
        if config.rescale_prenorm_residual:
            self._rescale_output_projections()
    
    def _init_weights(self, module):
        """
        Initialize weights for high accuracy training.
        
        Follows best practices from GPT-2, Mamba, and other SSM papers:
        - Linear layers: Normal distribution with std=base_std
        - Conv1d: Kaiming initialization for better gradient flow
        - LayerNorm/RMSNorm: weight=1, bias=0
        - Embeddings: Normal distribution with std=base_std
        """
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Conv1d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
    
    def _rescale_output_projections(self):
        """
        Rescale output projection weights by 1/sqrt(2*n_layers).
        
        This helps with training stability in deep networks by ensuring
        the residual stream variance doesn't grow with depth.
        (From GPT-2 and Mamba papers)
        """
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            # Scale the output projection of each Mamba block
            if hasattr(layer.mixer, 'out_proj'):
                with torch.no_grad():
                    layer.mixer.out_proj.weight.mul_(scale)
    
    def forward(self, x, inference_params=None):
        """
        Forward pass through all layers
        
        Args:
            x: (B, L, D) - input sequence
            inference_params: Optional inference parameters for caching
        
        Returns:
            (B, L, D) - output sequence
        """
        for layer in self.layers:
            x = layer(x, inference_params)
        
        # Apply final normalization
        x = self.final_norm(x)
        return x
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate inference cache for all layers
        
        Returns:
            List of (conv_state, ssm_state, velocity_state) for each layer
        """
        return [
            layer.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
            for layer in self.layers
        ]


class ResidualBlock(nn.Module):
    """
    Residual block: x + Mamba(RMSNorm(x))
    """
    def __init__(self, config: MuonMambaConfig, layer_idx: int = None):
        super().__init__()
        
        # Use the Mamba class from mamba_simple.py
        self.mixer = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            conv_bias=config.conv_bias,
            bias=config.bias,
            use_fast_path=config.use_fast_path,
            layer_idx=layer_idx,
            beta=config.momentum_beta,
            alpha=config.momentum_alpha,
            use_newton_schulz=config.use_newton_schulz,
            ns_steps=config.ns_steps,
        )
        
        # Use Triton RMSNorm if available, otherwise use fallback
        if TritonRMSNorm is not None:
            self.norm = TritonRMSNorm(config.d_model, eps=config.rms_norm_eps)
        else:
            self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
    
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (B, L, D)
            inference_params: Optional inference parameters
        Returns:
            (B, L, D)
        """
        return self.mixer(self.norm(x), inference_params) + x


# Convenience function to create MuonMamba from simple parameters
def create_muon_mamba(
    d_model: int,
    n_layers: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    beta: float = 0.9,
    alpha: float = 1.0,
    use_newton_schulz: bool = True,
    ns_steps: int = 1,
    base_std: float = 0.02,
    **kwargs
) -> MuonMamba:
    """
    Create a MuonMamba model with the given parameters.
    
    Args:
        d_model: Model dimension
        n_layers: Number of layers
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        beta: Momentum decay factor (default: 0.9)
        alpha: Momentum scale factor (default: 1.0)
        use_newton_schulz: Enable Newton-Schulz orthogonalization (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 1)
        base_std: Base standard deviation for initialization (default: 0.02)
        **kwargs: Additional config parameters
    
    Returns:
        MuonMamba model
    """
    config = MuonMambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand,
        momentum_beta=beta,
        momentum_alpha=alpha,
        use_newton_schulz=use_newton_schulz,
        ns_steps=ns_steps,
        base_std=base_std,
        **kwargs
    )
    return MuonMamba(config)
