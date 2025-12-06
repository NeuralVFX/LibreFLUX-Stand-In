import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import copy
import math

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from optimum.quanto import freeze, quantize, qfloat8, qint8, qint4, qint2, QTensor
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import AutoProcessor, SiglipVisionModel
from transformers import T5TokenizerFast, T5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

                                                                                                                 
import torch     
import torch.optim.lr_scheduler as tls
from torch.utils.data import Dataset                                                                              
import os                                                                                                         
import json                                                                                                       
import glob                                                                                                       
from PIL import Image                                                                                             
import numpy as np                                                                                                
from torchvision import transforms                                                                                
import random                                                                                                     
import cv2                                                                                                        


from ip_adapter.flux_ip_adapter import *
from ip_adapter.utils import is_torch2_available
from ip_adapter.flux_custom_pipelines import *

from models.transformer import *
from models import encode_prompt_helper


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gen_validation_images(pipe, test_dataloader, save_dir, iter, res):
    image_list = []
    input_image_list = []

    pipe.ip_adapter.eval() 

    for step, batch in enumerate(test_dataloader):
        pixel_values = batch["face_images"][0]
        
        input_img = (pixel_values.permute(1, 2, 0).cpu().numpy()+1)/2 if isinstance(pixel_values, torch.Tensor) else pixel_values
        input_image_list.append(input_img)

        images = pipe(
            prompt=batch['text'][0],
            negative_prompt="blurry",
            return_dict=False,
            num_inference_steps=75, # Add control for step count
            ref_adapter_image=pixel_values, 
            ref_adapter_scale=1.0,
            height=res,
            width=res, 
            generator = torch.Generator(device="cuda").manual_seed(19005)
        )
        
        image_list.append(images[0][0])
    
    # Create subplot grid: 2 rows (input + output)
    n_images = len(image_list)
    cols = n_images
    
    fig, axes = plt.subplots(2, cols, figsize=(cols*4, 8))
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    for idx in range(n_images):
        # Top row: input images
        axes[0, idx].imshow(input_image_list[idx])
        axes[0, idx].axis('off')
        title = axes[0, idx].set_title('Adapter Input', fontsize=10, color='white')
        title.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
        axes[0, idx].set_facecolor('none')  # Transparent subplot background
        
        # Bottom row: generated images
        axes[1, idx].imshow(image_list[idx])
        axes[1, idx].axis('off')
        title = axes[1, idx].set_title('Output', fontsize=10, color='white')
        title.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
        axes[1, idx].set_facecolor('none')  # Transparent subplot background
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/val.{iter:07d}.png", transparent=True)
    plt.close()

    pipe.ip_adapter.train()


class VFHQIpDataset(Dataset):                                                                                   
     def __init__(self,                                                                                            
                  json_dir,                                                                                        
                  face_size=512,                                                                                   
                  full_size=1024,                                                                                  
                  t_drop_rate=0.01,                                                                                
                  i_drop_rate=0.01,                                                                                
                  ti_drop_rate=0.01, 
                  lock=False,                                                                              
                  buffer=1.05):                                                                                    
                                                                                                                   
         super().__init__()                                                                                        
                                                                                                                   
         self.json_dir = json_dir                                                                                  
         self.face_size = face_size                                                                                
         self.full_size = full_size                                                                                
         self.t_drop_rate = t_drop_rate                                                                            
         self.i_drop_rate = i_drop_rate                                                                            
         self.ti_drop_rate = ti_drop_rate   
         self.lock = lock                                                                       
         self.buffer = buffer                                                                                      
                                                                                                                   
         # Load all JSON files                                                                                     
                                                                                                                   
         # Flatten all samples into a single list                                                                  
         self.samples = []
         jsonl_path = os.path.join(self.json_dir ,'dataset.jsonl' ) 
                                                                                                                          
         with open(jsonl_path, 'r') as f:                                                                              
            for line in f:                                                                                            
                self.samples.append(json.loads(line.strip()))                                                          
                                                                                                                   
         print(f" Loaded {len(self.samples)} sample JSON files")                         
                                                                                                                   
         # Define transforms                                                                                       
         self.face_transform = transforms.Compose([                                                                
             transforms.Resize((face_size, face_size), interpolation=transforms.InterpolationMode.BILINEAR),       
             transforms.ToTensor(),                                                                                
             transforms.Normalize([0.5], [0.5]),                                                                   
         ])                                                                                                        
                                                                                                                   
         self.full_transform = transforms.Compose([                                                                
             transforms.Resize((full_size, full_size), interpolation=transforms.InterpolationMode.BILINEAR),       
             transforms.ToTensor(),                                                                                
             transforms.Normalize([0.5], [0.5]),                                                                   
         ])                                                                                                        
                                                                                                                   
     def __len__(self):                                                                                            
         return len(self.samples)                                                                                  
                                                                                                                   
     def __getitem__(self, idx):      
         data_dict = self.samples[idx]
   

         # Select two different indices 
         if self.lock:
          full_img_choice=0   
          face_img_choice=len((data_dict))-1   
         else:                                                                     
          choices = [int(i) for i in range(len(data_dict))]
          full_img_choice = random.choice(choices)
          choices.pop(full_img_choice)
          face_img_choice = random.choice(choices) 

         full_dict = data_dict[full_img_choice]                                                                        
         face_dict = data_dict[face_img_choice]                                                                        
                                                                                                                   
         # Load images                                                                                             
         full_image_path = os.path.join(self.json_dir, full_dict['full_path'])                                     
         face_image_path = os.path.join(self.json_dir, face_dict['crop_path'])                                     
                                                                                                                   
         full_image = Image.open(full_image_path).convert('RGB')                                                   
         face_image = self.remove_alpha_with_grey_bg(Image.open(face_image_path))                                  
                                                                                                                   
         # Apply random crop to full image                                                                         
         full_image_np = np.array(full_image)                                                                      
         crop_coords = self.get_random_crop(                                                                       
             full_image_np,                                                                                        
             full_dict['crop_x'],                                                                                  
             full_dict['crop_y'],                                                                                  
             self.buffer                                                                                           
         )                                                                                                         
         cropped_full = full_image.crop((crop_coords[0], crop_coords[2],                                           
                                         crop_coords[1], crop_coords[3]))                                          
                                                                                                                   
         # Transform                                                                                               
         full_tensor = self.full_transform(cropped_full)                                                           
         face_tensor = self.face_transform(face_image)                                                             
                                                                                                                   
         # Text and dropout logic                                                                                  
         text = full_dict['caption']                                          
         drop_image_embed = 0                                                                                      
         rand_num = random.random()                                                                                
                                                                                                                   
         if rand_num < self.i_drop_rate:                                                                           
             drop_image_embed = 1                                                                                  
         elif rand_num < (self.i_drop_rate + self.t_drop_rate):                                                    
             text = ""                                                                                             
         elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):                                
             text = ""                                                                                             
             drop_image_embed = 1                                                                                  
                                                                                                                   
         return {                                                                                                  
             "image": full_tensor,                                                                                 
             "text": text,                                                                                         
             "face_image": face_tensor,                                                                            
             "drop_image_embed": drop_image_embed,                                                                 
         }                                                                                                         
                                                                                                                   
     def get_random_crop(self, image, crop_x, crop_y, buffer=1.05):                                                
         x_shape = image.shape[1]                                                                                  
         y_shape = image.shape[0]                                                                                  
         smallest_edge = min(y_shape, x_shape)                                                                     
         face_width = crop_x[1] - crop_x[0]                                                                        
         min_size = min(int(face_width * buffer), smallest_edge)                                                   
         crop_size = random.randint(min_size, smallest_edge)                                                       
         radius = crop_size // 2                                                                                   
                                                                                                                   
         min_x = max(radius, crop_x[1] - radius)                                                                   
         max_x = min(x_shape - radius, crop_x[0] + radius)                                                         
         min_y = max(radius, crop_y[1] - radius)                                                                   
         max_y = min(y_shape - radius, crop_y[0] + radius)                                                         
                                                                                                                   
         rand_x = random.randint(min_x, max_x)                                                                     
         rand_y = random.randint(min_y, max_y)                                                                     
                                                                                                                   
         return rand_x - radius, rand_x + radius, rand_y - radius, rand_y + radius                                 
                                                                                                                   
     def remove_alpha_with_grey_bg(self, image, grey_value=128):                                                   
         if image.mode == 'RGBA':                                                                                  
             background = Image.new('RGB', image.size, (grey_value, grey_value, grey_value))                       
             background.paste(image, mask=image.split()[3])                                                        
             return background                                                                                     
         return image.convert('RGB')     


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text = [example["text"] for example in data]
    face_images = torch.stack([example["face_image"] for example in data])
    
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    for i, drop in enumerate(drop_image_embeds):                                                             
        if drop == 1:                                                                                                 
            face_images[i] = torch.zeros_like(face_images[i])  

    return {
        "images": images,
        "face_images":face_images,
        "text": text,
        "drop_image_embeds": drop_image_embeds,
    }



def build_cos(
    optimizer,
    warmup_num_steps: int,
    cycle_first_step_size: int,
    cycle_mult: int = 1,
    floor_ratio: float = 0.1,
    last_batch_iteration: int = -1,
):
    peak_lrs  = [g["lr"] for g in optimizer.param_groups]
    floor_lrs = [lr * floor_ratio for lr in peak_lrs]

    class CosineWithRestarts:  # no inheritance
        def __init__(self):
            self.opt       = optimizer
            self.warmup    = warmup_num_steps
            self.base_len  = cycle_first_step_size
            self.mult      = max(1, cycle_mult)
            self.peak_lrs  = peak_lrs
            self.floor_lrs = floor_lrs
            self.last_step = last_batch_iteration

        def state_dict(self):
            return {"last_step": self.last_step}

        def load_state_dict(self, sd):
            self.last_step = sd.get("last_step", -1)

        def get_last_lr(self):
            return [g["lr"] for g in self.opt.param_groups]

        def step(self, step: int | None = None):
            # 1) figure out the current global step
            if step is None:
                step = self.last_step + 1
            self.last_step = step

            # 2) compute lr for each param-group
            if step < self.warmup:
                k = step / max(1, self.warmup)
                lrs = [f + (p - f) * k for p, f in zip(self.peak_lrs, self.floor_lrs)]
            else:
                step -= self.warmup
                cycle_len = self.base_len
                while step >= cycle_len:
                    step -= cycle_len
                    cycle_len *= self.mult
                phase = 0.5 * (1 + math.cos(2 * math.pi * step / cycle_len))
                lrs   = [f + (p - f) * phase for p, f in zip(self.peak_lrs, self.floor_lrs)]

            # 3) apply to optimizer
            for pg, lr in zip(self.opt.param_groups, lrs):
                pg["lr"] = lr
            return lrs

    return CosineWithRestarts()

def WarmupCosineRestartsLR(optimizer, **params):
    return build_cos(optimizer, **params)

tls.WarmupCosineRestartsLR = WarmupCosineRestartsLR

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ref_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Test data root path",
    )
    parser.add_argument(
        "--val_data_root_path",
        type=str,
        default="",
        required=True,
        help="Validation data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--ip_resolution",
        type=int,
        default=512,
        help=(
            "The resolution for ref images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=2000,
        help=(
            "Generate a validation image every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        # default="logit_normal",
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )   
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize everything except the adapter?",
    )
    parser.add_argument(
        "--cosine_lr",
        action="store_true",
        help="Cosine LR?",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
  
    #################################
    # Pipeline Loading/Assembly
    #################################
    revision = None
    variant = None

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )

    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )

    transformer = LibreFluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=revision,
        variant=variant,
    )


    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)


    transformer.eval()
    vae.eval()
    text_encoder_one.eval()
    text_encoder_two.eval()

    global_step = 0
    
    # To be used for training, and saving and loading weights
    if args.pretrained_ref_adapter_path is not None:
        
        ip_adapter = LibreFluxStandInIPAdapter(transformer,
                                        checkpoint=args.pretrained_ref_adapter_path)
        try:
            global_step = int( args.pretrained_ref_adapter_path.split('-')[-1].split('.')[0])
            print (f'Resuming at Global Step: {global_step}')

        except:
            print ('Couldnt Detect Global Step from pretrained_ip_adapter_path, starting from zero')
    else:
        ip_adapter = LibreFluxStandInIPAdapter(transformer)

    ip_adapter.train()

    pipeline = LibreFluxIpAdapterPipeline(
            scheduler=noise_scheduler,
            vae=vae,
            text_encoder=text_encoder_one,
            tokenizer=tokenizer_one,
            text_encoder_2 =text_encoder_two,
            tokenizer_2=tokenizer_two,
            transformer=transformer,
            ip_adapter=ip_adapter,
    )
    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set dtype
    ip_adapter.to(dtype=weight_dtype)
    transformer.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    text_encoder_one.to(dtype=weight_dtype)
    text_encoder_two.to(dtype=weight_dtype)

    # Quantize to save ram
    if args.quantize:
            # https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md
            # "Alternatively, you can go ham on quantisation here and run them [text encoders] in
            # int4 or int8 mode, because no one can stop you.""
            # Saves about 5GB
            print ("QUANTIZE: Base models except transformer...")


            # Transformer cant quantize, or backward pass in ip adapter breaks
            #quantize(transformer, weights=qint8)
            quantize(text_encoder_one, weights=qint8)
            quantize(text_encoder_two, weights=qint8)
            quantize(vae, weights=qint8)
            #freeze(transformer)
            freeze(text_encoder_one)
            freeze(text_encoder_two)
            freeze(vae)

            print ("Finished quantization")

    # Move to out device
    transformer.to(accelerator.device) 
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    ip_adapter.to(accelerator.device)


    #################################
    # Training Prep
    #################################

    # optimizer
    optimizer = torch.optim.AdamW(ip_adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  
    # scheduler
    scheduler = WarmupCosineRestartsLR(
    optimizer,
    warmup_num_steps=int(250),  # e.g. 5% of steps for warmup
    cycle_first_step_size=int(args.val_steps),  # length of first cosine cycle
    cycle_mult=1,          # keep cycles same length; tune if you like
    floor_ratio=0.1,       # minimum lr = 10% of peak
    last_batch_iteration=global_step,
    )

    train_dataset = VFHQIpDataset(                                                                                  
        json_dir=args.data_root_path,                  
        face_size=args.ip_resolution,             
        full_size=args.resolution,                                                                                                                                                                                    
    )                                                                                                                 
                                                                                                                      
    train_dataloader = torch.utils.data.DataLoader(                                                                   
        train_dataset,                                                                                                
        shuffle=True,                                                                                                 
        collate_fn=collate_fn,                                                                                        
        batch_size=args.train_batch_size,                                                                             
        num_workers=args.dataloader_num_workers,                                                                      
    )                                                                                                                 
                                                                                                                      
    val_dataset = VFHQIpDataset(                                                                                    
        json_dir=args.val_data_root_path,                   
        face_size=args.ip_resolution,                                                                                                                                                        
        full_size=args.resolution,   
        t_drop_rate=0.0, 
        i_drop_rate=0.0, 
        ti_drop_rate=0.0,   
        lock=True                                                                              
                                                                                                  
    )                                                                                                                 
                                                                                                                      
    val_dataloader = torch.utils.data.DataLoader(                                                                     
        val_dataset,                                                                                                  
        shuffle=False,                                                                                                
        collate_fn=collate_fn,                                                                                        
        batch_size=1,                                                                                                 
        num_workers=args.dataloader_num_workers,                                                                      
    )                                                                                                                 
                  

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    

    #################################
    # Training Loop
    #################################

    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):


                pixel_values = batch["images"]
                ref_pixel_values = batch["face_images"]
                current_lr = optimizer.param_groups[0]["lr"]

                with torch.no_grad():

                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        text_ids,
                        prompt_mask,
                    ) = encode_prompt_helper.encode_prompt_standalone(
                        prompt=batch['text'],
                        tokenizer_one=tokenizer_one,
                        text_encoder_one=text_encoder_one,
                        tokenizer_two=tokenizer_two,
                        text_encoder_two=text_encoder_two,
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                    )

                    ################
                    # Prepare empty text 
                    ################
                    (
                    null_prompt_embeds,
                    null_pooled_embeds,
                    null_text_ids,
                    null_prompt_mask,
                    ) = encode_prompt_helper.encode_prompt_standalone(
                        prompt="",
                        tokenizer_one=tokenizer_one,
                        text_encoder_one=text_encoder_one,
                        tokenizer_two=tokenizer_two,
                        text_encoder_two=text_encoder_two,
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                    )
                    joint_attention_kwargs = {"null_pooled_projections": null_pooled_embeds} # <--- Pass here

                    #################################
                    # Prepare Noisy Image
                    #################################   

                    model_input  = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                    
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))


                    latent_image_ids = LibreFluxIpAdapterPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2],
                        model_input.shape[3],
                        accelerator.device,
                        weight_dtype,
                    )

                    ##################################
                    # Prepare Stand In Ref
                    ##################################

                    
                    ref_model_input  = vae.encode(ref_pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                    
                    ref_model_input = (
                        ref_model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))


                    ref_image_ids = LibreFluxIpAdapterPipeline._prepare_ref_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2],
                        model_input.shape[3],
                        ref_model_input.shape[2],
                        ref_model_input.shape[3],
                        accelerator.device,
                        weight_dtype,
                    )

                    #################################
                    # End -  Prepare Stand In Ref
                    ################################

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=model_input.device
                    )

                    # 4. Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(
                        timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    
                    packed_noisy_model_input = LibreFluxIpAdapterPipeline._pack_latents(
                            noisy_model_input,
                            batch_size=model_input.shape[0],
                            num_channels_latents=model_input.shape[1],
                            height=model_input.shape[2],
                            width=model_input.shape[3],
                        )

                    #########################
                    # Pack Stand In Ref
                    #########################
                                        
                    packed_ref_model_input = LibreFluxIpAdapterPipeline._pack_latents(
                            ref_model_input,
                            batch_size=ref_model_input.shape[0],
                            num_channels_latents=ref_model_input.shape[1],
                            height=ref_model_input.shape[2],
                            width=ref_model_input.shape[3],
                        )

                    #########################
                    # End Pack Stand In Ref
                    #########################


                #################################
                # Forward Pass Through IP Adapter
                #################################
                guidance = None

                timesteps = (timesteps / 1000.0)
                text_ids = [ t for t in text_ids ]


                model_pred = ip_adapter(
                    packed_ref_model_input,
                    packed_noisy_model_input, # Added stand in ref input
                    timestep=timesteps,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    attention_mask=prompt_mask,
                    txt_ids=text_ids[0],
                    img_ids=latent_image_ids[0],
                    ref_img_ids=ref_image_ids[0], # Added stand in ref input
                    joint_attention_kwargs = joint_attention_kwargs,
                    return_dict=False,
                )[0]

                model_pred = LibreFluxIpAdapterPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor )//2,
                    width=int(model_input.shape[3] * vae_scale_factor )//2,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()


                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                if args.cosine_lr:
                    scheduler.step(global_step)

                optimizer.zero_grad()
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                if accelerator.is_main_process:
                    if global_step % 10 == 0:
                        print(
                            "Epoch {}, epoch_step {} global_step {}, data_time: {}, time: {}, "
                            "step_loss: {}, lr: {:.6f}".format(
                                epoch,
                                step,
                                global_step,
                                load_data_time,
                                time.perf_counter() - begin,
                                avg_loss,
                                current_lr,
                            )
                        )
            ###############################################
            # Validation and Saving ( not multi gpu compatible )
            ###############################################    
            global_step += 1
            
            if global_step % args.save_steps == 0:
                #save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #accelerator.save_state(save_path)
                unwrapped_model = accelerator.unwrap_model(ip_adapter)
                save_path = os.path.join(args.output_dir,
                                         f"checkpoint-{global_step:07d}.pt")
                unwrapped_model.save_pretrained(save_path)
                           
            if global_step % args.val_steps == 0:
              with torch.no_grad():
                gen_validation_images(pipeline,
                                      val_dataloader,
                                      args.output_dir,
                                      global_step,
                                      args.resolution)

            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    



