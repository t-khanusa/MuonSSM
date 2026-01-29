"""
HuggingFace-compatible wrapper for MuonLonghorn model.
This wrapper enables using the trained model with HuggingFace ecosystem tools
like lm-eval-harness, transformers generation utilities, etc.

Usage:
    from muon_longhorn_hf import MuonLonghornForCausalLM, MuonLonghornConfig
    
    model = MuonLonghornForCausalLM.from_pretrained("path/to/converted/model")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    
    # Generation
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
"""

import os
import json
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_online7_fn, selective_scan_online_orth_fn
except ImportError:
    selective_scan_online7_fn, selective_scan_online_orth_fn = None, None


class RMSNormFallback(nn.Module):
    """RMSNorm implementation - fallback if Triton version not available"""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MuonLonghornConfig(PretrainedConfig):
    """Configuration for MuonLonghorn model in HuggingFace format."""
    
    model_type = "muon_longhorn"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2304,
        num_hidden_layers: int = 11,
        num_attention_heads: int = 12,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        # MuonLonghorn specific
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        beta: float = 0.9,
        alpha: float = 1.0,
        use_newton_schulz: bool = True,
        ns_steps: int = 1,
        ns_mode: str = "compile",
        # Architecture flags
        use_swiglu: bool = True,
        rotary_percentage: float = 1.0,
        local_window: int = 2048,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        
        # MuonLonghorn specific
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.beta = beta
        self.alpha = alpha
        self.use_newton_schulz = use_newton_schulz
        self.ns_steps = ns_steps
        self.ns_mode = ns_mode
        
        # Architecture
        self.use_swiglu = use_swiglu
        self.rotary_percentage = rotary_percentage
        self.local_window = local_window
        
        # Computed
        self.d_inner = expand * hidden_size
        if dt_rank == "auto":
            self.dt_rank_value = math.ceil(hidden_size / 16)
        else:
            self.dt_rank_value = int(dt_rank)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MuonLonghornMixer(nn.Module):
    """MuonLonghorn mixer layer - SSM with momentum and Newton-Schulz."""
    
    def __init__(self, config: MuonLonghornConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank_value
        self.beta = config.beta
        self.alpha = config.alpha
        self.use_newton_schulz = config.use_newton_schulz
        self.ns_steps = config.ns_steps
        self.ns_mode = config.ns_mode
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        
        self.activation = "silu"
        self.act = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_head = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor:
        batch, seqlen, dim = hidden_states.shape
        
        # Project to inner dimension
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        
        x, z = xz.chunk(2, dim=1)
        
        # Convolution
        if causal_conv1d_fn is not None:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            x = self.act(self.conv1d(x)[..., :seqlen])
        
        # SSM projection
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, k, q = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_head.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        k = rearrange(k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        q = rearrange(q, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # SSM scan with momentum
        if self.use_newton_schulz and selective_scan_online_orth_fn is not None:
            y = selective_scan_online_orth_fn(
                x,
                q.to(x),
                k.to(x),
                dt.to(x),
                D=self.D.float(),
                t_bias=self.dt_head.bias.float(),
                z=z,
                return_last_state=False,
                beta=self.beta,
                alpha=self.alpha,
                ns_steps=self.ns_steps,
                ns_mode=self.ns_mode,
            )
        elif selective_scan_online7_fn is not None:
            y = selective_scan_online7_fn(
                x,
                q.to(x),
                k.to(x),
                dt.to(x),
                D=self.D.float(),
                t_bias=self.dt_head.bias.float(),
                z=z,
                return_last_state=False,
                beta=self.beta,
                alpha=self.alpha,
            )
        else:
            raise ImportError("SSM scan functions not available. Please install mamba_ssm.")
        
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer."""
    
    def __init__(self, config: MuonLonghornConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MuonLonghornBlock(nn.Module):
    """MuonLonghorn block: LayerNorm -> MuonLonghorn -> Residual -> LayerNorm -> MLP -> Residual"""
    
    def __init__(self, config: MuonLonghornConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization
        if RMSNorm is not None:
            self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm_1 = RMSNormFallback(config.hidden_size, eps=config.rms_norm_eps)
            self.norm_2 = RMSNormFallback(config.hidden_size, eps=config.rms_norm_eps)
        
        # MuonLonghorn attention
        self.attn = MuonLonghornMixer(config, layer_idx=layer_idx)
        
        # MLP
        self.mlp = SwiGLUMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # MuonLonghorn attention
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (None,)  # No KV cache for SSM
        
        if output_attentions:
            outputs += (None,)  # No attention weights for SSM
        
        return outputs


class MuonLonghornModel(PreTrainedModel):
    """MuonLonghorn model (backbone without LM head)."""
    
    config_class = MuonLonghornConfig
    base_model_prefix = "model"
    
    def __init__(self, config: MuonLonghornConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MuonLonghornBlock(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        if RMSNorm is not None:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = RMSNormFallback(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return (hidden_states, all_hidden_states, all_self_attns)


class MuonLonghornForCausalLM(PreTrainedModel, GenerationMixin):
    """MuonLonghorn model with LM head for causal language modeling."""
    
    config_class = MuonLonghornConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: MuonLonghornConfig):
        super().__init__(config)
        self.model = MuonLonghornModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # SSM doesn't use KV cache
            hidden_states=outputs[1] if output_hidden_states else None,
            attentions=outputs[2] if output_attentions else None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # SSM models process full sequence each time (no KV cache)
        model_inputs = {"input_ids": input_ids}
        model_inputs["attention_mask"] = attention_mask
        return model_inputs


# Register the model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

try:
    AutoConfig.register("muon_longhorn", MuonLonghornConfig)
    AutoModel.register(MuonLonghornConfig, MuonLonghornModel)
    AutoModelForCausalLM.register(MuonLonghornConfig, MuonLonghornForCausalLM)
except Exception:
    pass  # Already registered
