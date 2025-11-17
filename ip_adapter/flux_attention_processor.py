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


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)


class IPFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, num_heads=0):
        super().__init__()

        self.hidden_size = hidden_size 
        self.cross_attention_dim = cross_attention_dim 
        self.scale = scale
        self.num_tokens = num_tokens

        #self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        #self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.q_lora = LoRALinearLayer(hidden_size, hidden_size, rank=128)
        self.k_lora = LoRALinearLayer(hidden_size, hidden_size, rank=128)
        self.v_lora = LoRALinearLayer(hidden_size, hidden_size, rank=128)

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
        ref_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        ip_hidden_states = ip_encoder_hidden_states
        
        # `sample` projections.

        ###################################
        # Process latent and ref sep
        ###################################
        # Cut represents the number of ref tokens

        ref_hidden_states = None
        if ref_size is not None: 
          mod_cut = hidden_states.shape[1]-ref_size  # ( This works because the text is concatted on side A on the doulbe bloack )
          ref_hidden_states = hidden_states[:,mod_cut:]
          hidden_states = hidden_states[:,:mod_cut]

        query = attn.to_q(hidden_states) 
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        #########################
        # Run ref projection with LoRA
        #########################
        if ref_size is not None: 
          ref_query = attn.to_q(ref_hidden_states)+ self.q_lora(ref_hidden_states)
          ref_key = attn.to_k(ref_hidden_states)+ self.k_lora(ref_hidden_states)
          ref_value = attn.to_v(ref_hidden_states)+ self.v_lora(ref_hidden_states)

        #########################
        # End ref projection
        #########################
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ########################
        # Ref reshape
        #########################
        if ref_size is not None: 
          ref_query = ref_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
          ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
          ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
          #######################
        # End ref shape
        ######################

        #####################################
        # end change
        ###################################

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        #########################
        # Ref norm 
        ##########################
        if ref_size is not None: 
            ref_query = attn.norm_q(ref_query) #### normalizing ref
            ref_key = attn.norm_k(ref_key) #### normalizing ref

        # handle IP attention FIRST
        #####################################
        # IP Adapter not used in Stand -In training - kept for compatibility
        #####################################
        # for ip-adapter
        """
        if ip_hidden_states != None:
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
        """
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

        #######################
        # Prep Rotary Emb
        #######################
        if ref_size is not None:
            ref_rotary_emb = (image_rotary_emb[0][-ref_size:], image_rotary_emb[1][-ref_size:])
            image_rotary_emb = (image_rotary_emb[0][:-ref_size], image_rotary_emb[1][:-ref_size])
            
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        ##################
        # Ref Rotary Emb
        ##################
        if ref_size is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            ref_query = apply_rotary_emb(ref_query, ref_rotary_emb)
            ref_key = apply_rotary_emb(ref_key, ref_rotary_emb)
        ###################
        # End Ref Rotary
        ###################

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (attention_mask > 0).bool()
            attention_mask = attention_mask.to(
                device=hidden_states.device, dtype=query.dtype
            )
        original_hidden_states = hidden_states

        #######################################
        # Ref Query
        #######################################

        # ref query doesnt need attention masking, casue no prompt
        if ref_size is not None:
          ref_hidden_states = F.scaled_dot_product_attention(
              ref_query,
              ref_key,
              ref_value,
              dropout_p=0.0,
              is_causal=False,
          )

          ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(
              batch_size, -1, attn.heads * head_dim
          )
          ref_hidden_states = ref_hidden_states.to(query.dtype)

        ##################
        # End Ref Query
        ##################
        

        # both query
        #######################################
        # Cat Ref if needed
        #######################################
        if ref_size is not None:
            layer_scale_expanded = layer_scale.view(-1, 1, 1, 1)
            cat_key = torch.cat([key, layer_scale_expanded*ref_key], dim=2)
            cat_value = torch.cat([value, layer_scale_expanded*ref_value], dim=2)
            
        if attention_mask is not None:
            ref_mask_ext = torch.ones(attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], ref_size, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, ref_mask_ext], dim=-1)
        else:
            cat_key = key
            cat_value = value

        hidden_states = F.scaled_dot_product_attention(
            query, # This allows only the image, to also see the ref ( not both ways )
            cat_key,
            cat_value,
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
                    #if ip_hidden_states != None:
                    #  hidden_states = hidden_states + (self.scale * layer_scale) * ip_hidden_states

                    ########
                    # Catting states
                    #########
                    if ref_size is not None:
                      hidden_states = torch.cat([hidden_states,ref_hidden_states],dim=1)

                    # linear proj
                    hidden_states = attn.to_out[0](hidden_states)
                    # dropout
                    hidden_states = attn.to_out[1](hidden_states)
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states

        else:
            
                    # Final injection of ip addapter hidden_states
                    #if ip_hidden_states != None:
                    #  hidden_states = hidden_states + (self.scale * layer_scale) * ip_hidden_states

                    ########
                    # Catting states
                    #########
                    if ref_size is not None:
                      hidden_states = torch.cat([hidden_states,ref_hidden_states],dim=1)

                    if attn.to_out is not None:
                        hidden_states = attn.to_out[0](hidden_states)
                        hidden_states = attn.to_out[1](hidden_states)

                    return hidden_states