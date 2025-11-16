from itertools import chain
import torch
from torch import nn
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
)

from ip_adapter.flux_attention_processor import *


class ImageProjModel(nn.Module):
    def __init__(self, clip_dim=768, cross_attention_dim=4096, num_tokens=16):
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.clip_dim = clip_dim

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_dim,clip_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(clip_dim*2, cross_attention_dim*num_tokens),
        )        
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
    
    def forward(self,input):
        
        raw_proj = self.proj(input)
        reshaped_proj = raw_proj.reshape(input.shape[0],self.num_tokens,self.cross_attention_dim)
        reshaped_proj = self.norm( reshaped_proj )

        return reshaped_proj


class LibreFluxIPAdapter(nn.Module):
    def __init__(self, transformer, image_proj_model, checkpoint=None):
        super().__init__()
        self.transformer = transformer
        self.image_proj_model = image_proj_model

        # Using startswith uses only double transformer blocks, and skips the single transformer blocks
        self.culled_transformer_blocks = {}
        for name, module in self.transformer.named_modules():
            if isinstance(module, Attention):
                if name.startswith('transformer_blocks') or name.startswith('single_transformer_blocks'):
                    #print (f"Using Transformer: {name}")
                    self.culled_transformer_blocks[name] = module
                #else:
                    #print (f"Ignoring Transformer: {name}")
        # Apply the adapter to the culled blocks
        self.wrap_attention_blocks()
        
        if checkpoint:
            self.load_from_checkpoint(checkpoint)

    def wrap_attention_blocks(self,scale=1.0, num_tokens=16):
        """ Inject the IP-Adapter modules into the Transformer model """
        sample_attn = self.transformer.transformer_blocks[0].attn

        hidden_size = sample_attn.inner_dim
        cross_attention_dim = sample_attn.cross_attention_dim
        num_heads = sample_attn.heads
        scale = 1.0
        num_tokens = 16
 
        processor_list = []
        for name in self.culled_transformer_blocks:
            module = self.culled_transformer_blocks[name]
            print (f"Adding Attention IP Wrapper: {name}")
            module.processor = IPFluxAttnProcessor2_0(
                    hidden_size= hidden_size,
                    cross_attention_dim=4096,
                    num_heads=num_heads,
                    scale=1.0,
                    num_tokens=16,
                )
            processor_list.append(module.processor )

        # Store adapters as a module list for saving/loading
        self.adapter_modules = torch.nn.ModuleList(processor_list)
        
    def parameters(self):
        """ Easy way to return all params """
        # Apply adapter
        adapter_param_list = []
        for name in self.culled_transformer_blocks:
            module = self.culled_transformer_blocks[name]            
            adapter_param_list.append(module.processor.parameters())
                    
        all_params = chain(*adapter_param_list,self.image_proj_model.parameters())
        return all_params

    def forward(self, ref_image, *args, layer_scale= torch.Tensor([1.0]), **kwargs):
        """ Run projection and run forward """

        ip_encoder_hidden_states = self.image_proj_model(ref_image)

        # Add ip hidden states to kwargs
        if 'joint_attention_kwargs' not in kwargs:
            kwargs['joint_attention_kwargs'] = {}
        layer_scale = layer_scale.to(dtype=ip_encoder_hidden_states.dtype,
        device=ip_encoder_hidden_states.device)   

        kwargs['joint_attention_kwargs']['ip_layer_scale'] = layer_scale
        kwargs['joint_attention_kwargs']['ip_hidden_states'] = ip_encoder_hidden_states

        output = self.transformer(*args,
                **kwargs)

        return output

    def save_pretrained(self,ckpt_path):
        """ Save model weights """
        state_dict = {}

        state_dict["image_proj"] = self.image_proj_model.state_dict()
        state_dict["ip_adapter"] = self.adapter_modules.state_dict()
        torch.save(state_dict, ckpt_path)

    def load_from_checkpoint(self, ckpt_path):
        """ Loader ripped from tencent repo """
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    @property
    def dtype(self):
        return next(self.image_proj_model.parameters()).dtype

### Examples
# Test
#image_proj_model = ImageProjModel(clip_dim=768, cross_attention_dim=3072, num_tokens=16)
#emb =  torch.ones(2,768)
#image_proj_model(emb)

#image_proj_model = ImageProjModel(clip_dim=768, cross_attention_dim=3072, num_tokens=16)
#IP_Adatper = LibreFluxIPAdapter(transformer,image_proj_model)