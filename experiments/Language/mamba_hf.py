#!/usr/bin/env python3
"""
HuggingFace wrapper for Mamba architecture.

Mamba models use mamba_ssm library and have a different structure:
- Uses Mamba mixer from mamba_ssm.modules.mamba_simple
- Has norm_1, attn (Mamba), norm_2, mlp structure
- Similar to other architectures but uses Mamba SSM
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig
from transformers.generation import GenerationMixin

# Import Mamba from mamba_ssm
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    raise ImportError("mamba_ssm not installed. Please install it: pip install mamba-ssm")


class MambaConfig(PretrainedConfig):
    """Configuration for Mamba model in HuggingFace format."""
    
    model_type = "mamba"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 672,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 10,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        # Mamba specific
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        d_conv: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        # Architecture
        use_swiglu: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        
        # Mamba specific
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv if d_conv is not None else conv_kernel
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        
        # Architecture
        self.use_swiglu = use_swiglu
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class MambaBlock(nn.Module):
    """Mamba transformer block."""
    
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Create Mamba mixer
        self.mixer = Mamba(
            d_model=config.hidden_size,
            layer_idx=layer_idx,
            use_fast_path=False,
        )
        
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
        
        # Mixer (Mamba)
        hidden_states = self.mixer(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, None  # Mamba doesn't use past_key_values in the same way


class MambaModel(PreTrainedModel):
    """Mamba model (backbone without LM head)."""
    
    config_class = MambaConfig
    base_model_prefix = "model"
    
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Layers
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=i)
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
        
        all_hidden_states = () if output_hidden_states else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states] if v is not None)
        
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM(PreTrainedModel, GenerationMixin):
    """Mamba model with LM head for causal language modeling."""
    
    config_class = MambaConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.model = MambaModel(config)
        
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
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        # Mamba doesn't use traditional KV cache
        return None


# Register the model with transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

try:
    AutoConfig.register("mamba", MambaConfig)
    AutoModel.register(MambaConfig, MambaModel)
    AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM)
except Exception:
    pass  # Already registered
