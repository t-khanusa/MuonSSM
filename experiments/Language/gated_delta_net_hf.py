#!/usr/bin/env python3
"""
HuggingFace wrapper for GatedDeltaNet architecture.

GatedDeltaNet uses a different mixer architecture than MuonLonghorn:
- Uses chunk_gated_delta_rule operation
- Has q_proj, k_proj, v_proj, g_proj, gk_proj, b_proj
- Uses q_conv1d, k_conv1d, v_conv1d (ShortConvolution)
- Uses FusedRMSNormSwishGate or separate norm + gate
- Different state structure

This is a separate wrapper from MuonLonghorn!
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

# Import GatedDeltaNet from lit_gpt
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(wd))

from lit_gpt.gated_delta_net import GatedDeltaNet


class GatedDeltaNetConfig(PretrainedConfig):
    """Configuration for GatedDeltaNet model in HuggingFace format."""
    
    model_type = "gated_delta_net"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2304,
        num_hidden_layers: int = 11,
        num_attention_heads: int = 12,  # GatedDeltaNet default
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        # GatedDeltaNet specific
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        num_kv_heads: Optional[int] = None,
        qk_norm: str = "l2",
        conv_size: int = 4,
        gate_fn: str = "swish",
        use_mamba_gate: bool = True,
        use_residual: bool = False,
        use_input_gate: bool = False,
        fuse_norm: bool = True,
        # Architecture
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
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        
        # GatedDeltaNet specific
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.qk_norm = qk_norm
        self.conv_size = conv_size
        self.gate_fn = gate_fn
        self.use_mamba_gate = use_mamba_gate
        self.use_residual = use_residual
        self.use_input_gate = use_input_gate
        self.fuse_norm = fuse_norm
        
        # Architecture
        self.use_swiglu = use_swiglu
        self.rotary_percentage = rotary_percentage
        self.local_window = local_window
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class SimpleCache:
    """Simple cache adapter for GatedDeltaNet."""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self._cache = None
    
    def update(self, state, layer_idx, seq_len):
        """Update cache for this layer."""
        self._cache = state
    
    def __getitem__(self, idx):
        """Get cache for layer."""
        if idx == self.layer_idx:
            return self._cache
        return None


class GatedDeltaNetMixer(GatedDeltaNet):
    """
    GatedDeltaNet mixer layer for HuggingFace.
    
    This directly inherits from GatedDeltaNet, so weights map as:
    attn.q_proj -> mixer.q_proj
    """
    
    def __init__(self, config: GatedDeltaNetConfig, layer_idx: int = None):
        # Initialize GatedDeltaNet with config parameters
        super().__init__(
            mode='chunk',
            hidden_size=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            qk_norm=config.qk_norm,
            conv_size=config.conv_size,
            gate_fn=config.gate_fn,
            use_mamba_gate=config.use_mamba_gate,
            use_residual=config.use_residual,
            use_input_gate=config.use_input_gate,
            fuse_norm=config.fuse_norm,
            layer_idx=layer_idx,
        )
        self.config = config
        self._layer_idx = layer_idx
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """Forward pass - adapt cache format for GatedDeltaNet."""
        # Create a proper cache wrapper for GatedDeltaNet
        # If past_key_values is None but use_cache=True, we need to create cache
        cache = None
        if use_cache:
            cache = SimpleCache(self._layer_idx)
            if past_key_values is not None:
                if hasattr(past_key_values, '_cache'):
                    # Already a SimpleCache
                    cache._cache = past_key_values._cache
                elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                    # Raw tuple/list cache
                    cache._cache = past_key_values
                # else: cache._cache remains None (first step)
        
        # Call parent forward
        output, attention, updated_cache = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        
        # Extract cache from SimpleCache if needed
        if updated_cache is not None and hasattr(updated_cache, '_cache'):
            updated_cache = updated_cache._cache
        
        return output, attention, updated_cache


class RMSNorm(nn.Module):
    """RMSNorm layer."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP layer."""
    
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class GatedDeltaNetBlock(nn.Module):
    """GatedDeltaNet transformer block."""
    
    def __init__(self, config: GatedDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = GatedDeltaNetMixer(config, layer_idx=layer_idx)
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        
        # Mixer
        hidden_states, _, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, past_key_values


class GatedDeltaNetModel(PreTrainedModel):
    """GatedDeltaNet model (backbone without LM head)."""
    
    config_class = GatedDeltaNetConfig
    base_model_prefix = "model"
    
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__(config)
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Layers
        self.layers = nn.ModuleList([
            GatedDeltaNetBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Handle past_key_values - convert from various formats to our list format
        # HuggingFace may pass DynamicCache or other cache types that we don't support
        if past_key_values is not None:
            # Check if it's our list format
            if isinstance(past_key_values, (list, tuple)):
                past_key_values = list(past_key_values)  # Ensure it's mutable
            else:
                # Unknown cache type (e.g., DynamicCache) - extract if possible or ignore
                try:
                    # Try to convert to list if it supports indexing
                    past_key_values = [past_key_values[i] if i < len(past_key_values) else None 
                                       for i in range(len(self.layers))]
                except (TypeError, AttributeError):
                    # Can't convert - start fresh
                    past_key_values = [None] * len(self.layers)
        elif use_cache:
            past_key_values = [None] * len(self.layers)
        
        all_hidden_states = () if output_hidden_states else None
        new_past_key_values = [] if use_cache else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, layer_past = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            
            if use_cache:
                new_past_key_values.append(layer_past)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Return our list-based cache format
        final_past_key_values = tuple(new_past_key_values) if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, final_past_key_values, all_hidden_states] if v is not None)
        
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=final_past_key_values,
            hidden_states=all_hidden_states,
        )


class GatedDeltaNetForCausalLM(PreTrainedModel, GenerationMixin):
    """GatedDeltaNet model with LM head for causal language modeling."""
    
    config_class = GatedDeltaNetConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__(config)
        self.model = GatedDeltaNetModel(config)
        
        # LM head (tied to embeddings if configured)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens
        return self.lm_head
    
    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.model.embed_tokens = value
        else:
            self.lm_head = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        
        # LM head
        if self.config.tie_word_embeddings:
            logits = torch.nn.functional.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # If using past_key_values, only use the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # Don't pass attention_mask for cached generation
            # The single new token is always unmasked, and passing truncated mask
            # causes issues with convolution layers
            attention_mask = None
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past.append(None)
            else:
                # GatedDeltaNet cache structure
                reordered_past.append(tuple(p[beam_idx] for p in layer_past))
        return reordered_past


# Register the model with transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

try:
    AutoConfig.register("gated_delta_net", GatedDeltaNetConfig)
    AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
except Exception:
    pass  # Already registered
