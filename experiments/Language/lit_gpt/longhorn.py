import json
import math
import os
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from mamba_ssm.ops.selective_scan_interface import longhorn_inner_fn, selective_scan_online7_fn, selective_scan_online_orth_fn
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class LonghornConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class Longhorn(nn.Module):
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
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state*2, bias=False, **factory_kwargs
        )

        self.dt_head = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            out = longhorn_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_head.weight,
                None,
                D=self.D,
                delta_bias=self.dt_head.bias.float(),
                out_proj_weight=self.out_proj.weight,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_head.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]

            y = selective_scan_online7_fn(x,
                                          q.to(x),
                                          k.to(x),
                                          dt.to(x),
                                          D=self.D.float(),
                                          t_bias=self.dt_head.bias.float(),
                                          z=z,
                                          return_last_state=ssm_state is not None
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, k, q = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.linear(dt, self.dt_head.weight)  # (B d_inner)
        dt = torch.sigmoid(dt + self.dt_head.bias.to(dtype=dt.dtype))
        dt = dt / (1 + dt * k.square().sum(dim=-1, keepdim=True))

        dA = 1 - torch.einsum("bd,bn->bdn", dt, k.pow(2))
        dB = torch.einsum("bd,bn->bdn", dt, k)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), q)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


@dataclass
class MuonLonghornConfig:
    """Configuration for MuonLonghorn (Longhorn with momentum and Newton-Schulz)."""
    
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    # MuonLonghorn specific
    beta: float = 0.9  # Velocity decay (momentum)
    alpha: float = 1.0  # Velocity scale
    use_newton_schulz: bool = True
    ns_steps: int = 1
    ns_mode: str = 'compile'  # 'compile' or 'triton'


class MuonLonghorn(nn.Module):
    """MuonLonghorn: Longhorn with momentum velocity scan and Newton-Schulz orthogonalization.
    
    This class extends Longhorn with:
    - Two-stage scan: velocity state v_t = beta * v_{t-1} + alpha * input, hidden state h_t = forget * h_{t-1} + v_t
    - Newton-Schulz orthogonalization of the velocity input (dt * u * K)
    
    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        beta: Velocity decay factor (momentum), 0.0 = no momentum
        alpha: Velocity scale factor
        use_newton_schulz: Whether to apply Newton-Schulz orthogonalization
        ns_steps: Number of Newton-Schulz iterations
        ns_mode: 'compile' for torch.compile or 'triton' for Triton kernel
        dt_rank: Rank of delta projection, 'auto' = ceil(d_model / 16)
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Delta initialization mode
        dt_scale: Delta scale factor
        dt_init_floor: Floor for delta initialization
        conv_bias: Whether to use bias in convolution
        bias: Whether to use bias in linear layers
        use_fast_path: Whether to use fused kernel
        layer_idx: Layer index for caching
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        # MuonLonghorn specific
        beta=0.9,
        alpha=1.0,
        use_newton_schulz=True,
        ns_steps=1,
        ns_mode='compile',
        # Original Longhorn parameters
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # MuonLonghorn specific
        self.beta = beta
        self.alpha = alpha
        self.use_newton_schulz = use_newton_schulz
        self.ns_steps = ns_steps
        self.ns_mode = ns_mode

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state*2, bias=False, **factory_kwargs
        )

        self.dt_head = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_head.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]

        # MuonLonghorn: Use appropriate scan function based on newton_schulz setting
        if self.use_newton_schulz:
            y = selective_scan_online_orth_fn(
                x,
                q.to(x),
                k.to(x),
                dt.to(x),
                D=self.D.float(),
                t_bias=self.dt_head.bias.float(),
                z=z,
                return_last_state=ssm_state is not None,
                beta=self.beta,
                alpha=self.alpha,
                ns_steps=self.ns_steps,
                ns_mode=self.ns_mode,
            )
        else:
            # print("Using selective_scan_online7_fn")
            y = selective_scan_online7_fn(
                x,
                q.to(x),
                k.to(x),
                dt.to(x),
                D=self.D.float(),
                t_bias=self.dt_head.bias.float(),
                z=z,
                return_last_state=ssm_state is not None,
                beta=self.beta,
                alpha=self.alpha,
            )
        
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """Single step forward for inference/generation."""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, k, q = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.linear(dt, self.dt_head.weight)  # (B d_inner)
        dt = torch.sigmoid(dt + self.dt_head.bias.to(dtype=dt.dtype))
        dt = dt / (1 + dt * k.square().sum(dim=-1, keepdim=True))

        dA = 1 - torch.einsum("bd,bn->bdn", dt, k.pow(2))
        dB = torch.einsum("bd,bn->bdn", dt, k)
        
        # MuonLonghorn: Step-wise velocity update (simplified, without NS for step mode)
        if hasattr(self, '_velocity_state') and self._velocity_state is not None:
            velocity = self.beta * self._velocity_state + self.alpha * rearrange(x, "b d -> b d 1") * dB
            self._velocity_state = velocity
        else:
            velocity = self.alpha * rearrange(x, "b d -> b d 1") * dB
            self._velocity_state = velocity
        
        ssm_state.copy_(ssm_state * dA + velocity)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), q)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_head.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        # MuonLonghorn: Also allocate velocity state
        self._velocity_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_head.weight.device,
                dtype=self.dt_head.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Longhorn, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class LonghornLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: LonghornConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = LonghornConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for LonghornLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)


class LonghornLM(nn.Module):
    def __init__(self, config: LonghornConfig):
        super().__init__()
        config.ssm_cfg = {
            "d_state": config.d_state
        }
        self.transformer = LonghornLMHeadModel(config)

    def forward(self, idx, targets=None):
        logits = self.transformer(idx).logits

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = logits[:, [-1], :]
            loss = None
        return logits, loss


# =============================================================================
# MuonLonghorn Full Model Stack
# =============================================================================

def create_muon_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Create a Block with MuonLonghorn mixer instead of vanilla Longhorn."""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(MuonLonghorn, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MuonMixerModel(nn.Module):
    """MixerModel using MuonLonghorn blocks instead of vanilla Longhorn."""
    
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_muon_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MuonLonghornLMHeadModel(nn.Module, GenerationMixin):
    """Full MuonLonghorn language model with LM head, built from MuonLonghornConfig."""

    def __init__(
        self,
        config: MuonLonghornConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        # Build ssm_cfg from config's MuonLonghorn-specific parameters
        ssm_cfg = dict(config.ssm_cfg) if config.ssm_cfg else {}
        ssm_cfg.update({
            'beta': config.beta,
            'alpha': config.alpha,
            'use_newton_schulz': config.use_newton_schulz,
            'ns_steps': config.ns_steps,
            'ns_mode': config.ns_mode,
        })

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        
        self.backbone = MuonMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def save_pretrained(self, save_directory):
        """Save the model and its configuration to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f)


class MuonLonghornLM(nn.Module):
    """Wrapper class for MuonLonghornLMHeadModel with training loss computation."""
    
    def __init__(self, config: MuonLonghornConfig, device=None, dtype=None):
        super().__init__()
        self.transformer = MuonLonghornLMHeadModel(config, device=device, dtype=dtype)

    def forward(self, idx, targets=None):
        logits = self.transformer(idx).logits

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = logits[:, [-1], :]
            loss = None
        return logits, loss


# =============================================================================
# MuonLonghorn Stack (Similar to MuonMamba structure)
# =============================================================================

class RMSNormFallback(nn.Module):
    """RMSNorm implementation - fallback if Triton version not available"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


@dataclass
class MuonLonghornStackConfig:
    """
    Configuration for MuonLonghornStack: Multi-layer MuonLonghorn for feature extraction
    
    This is a simpler config (similar to MuonMambaConfig) without LM-specific parameters.
    Use this for feature extraction tasks, not language modeling.
    
    Key parameters:
    - beta: β ∈ [0, 1] - Controls velocity decay (0 = no momentum, 0.9 = high momentum)
    - alpha: α > 0 - Scales the velocity contribution (typically 0.5-1.5)
    - use_newton_schulz: When True, NS orthogonalization is applied to stabilize momentum
    """
    d_model: int  # D - model dimension
    n_layers: int  # Number of layers
    d_state: int = 16  # N - SSM state dimension
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor
    dt_rank: str = 'auto'  # Rank of delta projection
    
    # SSM discretization parameters
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    
    # MuonLonghorn: Momentum + Newton-Schulz parameters
    beta: float = 0.9  # β - momentum decay factor (0 = no momentum)
    alpha: float = 1.0  # α - momentum input scaling
    use_newton_schulz: bool = True  # Enable Newton-Schulz orthogonalization
    ns_steps: int = 1  # Number of Newton-Schulz iterations
    ns_mode: str = 'compile'  # 'compile' or 'triton'
    
    # Normalization
    rms_norm_eps: float = 1e-5
    
    # Architecture options
    bias: bool = False
    conv_bias: bool = True
    
    # Initialization parameters
    base_std: float = 0.02  # Base standard deviation for weight initialization
    rescale_prenorm_residual: bool = True  # Scale output projection by 1/sqrt(2*n_layers)
    
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        # Validate momentum parameters
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")


class MuonLonghornResidualBlock(nn.Module):
    """
    Residual block: x + MuonLonghorn(RMSNorm(x))
    
    Similar to MuonMamba's ResidualBlock but uses MuonLonghorn mixer.
    """
    def __init__(self, config: MuonLonghornStackConfig, layer_idx: int = None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Use MuonLonghorn mixer
        self.mixer = MuonLonghorn(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            conv_bias=config.conv_bias,
            bias=config.bias,
            layer_idx=layer_idx,
            beta=config.beta,
            alpha=config.alpha,
            use_newton_schulz=config.use_newton_schulz,
            ns_steps=config.ns_steps,
            ns_mode=config.ns_mode,
            **factory_kwargs,
        )
        
        # Use Triton RMSNorm if available, otherwise use fallback
        if RMSNorm is not None:
            self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)
        else:
            self.norm = RMSNormFallback(config.d_model, eps=config.rms_norm_eps)
            if device is not None:
                self.norm = self.norm.to(device)
            if dtype is not None:
                self.norm = self.norm.to(dtype)
    
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (B, L, D)
            inference_params: Optional inference parameters
        Returns:
            (B, L, D)
        """
        return self.mixer(self.norm(x), inference_params) + x


class MuonLonghornStack(nn.Module):
    """
    Multi-layer MuonLonghorn model with residual connections
    
    MuonLonghorn = Longhorn + Momentum + Newton-Schulz Orthogonalization
    
    Architecture:
    - Each layer: RMSNorm → MuonLonghorn → Residual connection
    - MuonLonghorn: Uses momentum-based SSM with optional NS orthogonalization
    
    Similar to MuonMamba but uses Longhorn's Q-K attention-like SSM.
    
    Example:
        config = MuonLonghornStackConfig(
            d_model=256,
            n_layers=4,
            d_state=16,
            beta=0.9,
            alpha=1.0,
            use_newton_schulz=True,
        )
        model = MuonLonghornStack(config, device='cuda')
        
        # Forward pass
        x = torch.randn(batch, seq_len, d_model, device='cuda')
        y = model(x)  # (batch, seq_len, d_model)
    """
    def __init__(self, config: MuonLonghornStackConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.config = config
        self.n_layers = config.n_layers
        
        self.layers = nn.ModuleList([
            MuonLonghornResidualBlock(config, layer_idx=i, **factory_kwargs) 
            for i in range(config.n_layers)
        ])
        
        # Final norm
        if RMSNorm is not None:
            self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)
        else:
            self.final_norm = RMSNormFallback(config.d_model, eps=config.rms_norm_eps)
            if device is not None:
                self.final_norm = self.final_norm.to(device)
            if dtype is not None:
                self.final_norm = self.final_norm.to(dtype)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections (scale by depth)
        if config.rescale_prenorm_residual:
            self._rescale_output_projections()
    
    def _init_weights(self, module):
        """
        Initialize weights for high accuracy training.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, RMSNormFallback)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _rescale_output_projections(self):
        """
        Rescale output projection weights by 1/sqrt(2*n_layers).
        
        This helps with training stability in deep networks.
        """
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
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
        """
        return [
            layer.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
            for layer in self.layers
        ]


# Convenience function to create MuonLonghornStack from simple parameters
def create_muon_longhorn(
    d_model: int,
    n_layers: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    beta: float = 0.9,
    alpha: float = 1.0,
    use_newton_schulz: bool = True,
    ns_steps: int = 1,
    ns_mode: str = 'compile',
    base_std: float = 0.02,
    device=None,
    dtype=None,
    **kwargs
) -> MuonLonghornStack:
    """
    Create a MuonLonghornStack model with the given parameters.
    
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
        ns_mode: Newton-Schulz mode - 'compile' or 'triton' (default: 'compile')
        base_std: Base standard deviation for initialization (default: 0.02)
        device: Device to create model on (default: None)
        dtype: Data type for model (default: None)
        **kwargs: Additional config parameters
    
    Returns:
        MuonLonghornStack model
        
    Example:
        model = create_muon_longhorn(
            d_model=256,
            n_layers=4,
            beta=0.9,
            use_newton_schulz=True,
            device='cuda',
            dtype=torch.float32,
        )
    """
    config = MuonLonghornStackConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        beta=beta,
        alpha=alpha,
        use_newton_schulz=use_newton_schulz,
        ns_steps=ns_steps,
        ns_mode=ns_mode,
        base_std=base_std,
        **kwargs
    )
    return MuonLonghornStack(config, device=device, dtype=dtype)