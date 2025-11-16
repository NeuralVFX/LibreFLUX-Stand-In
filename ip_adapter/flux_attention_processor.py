from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention
import inspect
from functools import partial
from diffusers.models.normalization import RMSNorm
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn


from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention
import inspect
from functools import partial
from diffusers.models.normalization import RMSNorm
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn


class IPFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, num_heads=0):
        super().__init__()

        self.hidden_size = hidden_size 
        self.cross_attention_dim = cross_attention_dim 
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        ip_encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        layer_scale: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        ip_hidden_states = ip_encoder_hidden_states
        
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # handle IP attention FIRST

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        # reshaping to match query shape
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_key = self.norm_added_k(ip_key)


        # Using flux stype attention here
        ip_hidden_states = F.scaled_dot_product_attention(
            query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=None,
        )

        # reshaping ip_hidden_states in the same way as hidden_states
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)

            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)

            key = apply_rotary_emb(key, image_rotary_emb)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (attention_mask > 0).bool()
            attention_mask = attention_mask.to(
                device=hidden_states.device, dtype=query.dtype
            )
        original_hidden_states = hidden_states

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attention_mask,
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)


        layer_scale = layer_scale.view(-1, 1, 1)

        if encoder_hidden_states is not None:

                    encoder_hidden_states, hidden_states = (
                        hidden_states[:, : encoder_hidden_states.shape[1]],
                        hidden_states[:, encoder_hidden_states.shape[1] :],
                    )
                
                    # Final injection of ip addapter hidden_states
                    hidden_states = hidden_states + (self.scale * layer_scale) * ip_hidden_states

                    # linear proj
                    hidden_states = attn.to_out[0](hidden_states)
                    # dropout
                    hidden_states = attn.to_out[1](hidden_states)
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states

        else:
            
                    # Final injection of ip addapter hidden_states
                    hidden_states = hidden_states + (self.scale * layer_scale) * ip_hidden_states

                    if attn.to_out is not None:
                        hidden_states = attn.to_out[0](hidden_states)
                        hidden_states = attn.to_out[1](hidden_states)

                    return hidden_states