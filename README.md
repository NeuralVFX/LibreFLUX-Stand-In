# LibreFLUX-IP-Adapter
![LibreFLUX-IP-Adapter example](examples/matrix_edge.png)

The intent of this repo is to make it possible to train an IP-Adapter based on LibreFLUX. This means:
- Incorporating Attention Masking
- Removing the distilled guidance vector during training
- Running inference with CFG
- Using IP-Adapter architecture for image-conditioned generation

**Disclaimer**: This is pieced together by modifying borrowing and referencing these repos:
- https://github.com/tencent-ailab/IP-Adapter (Training code forked from here)
- [https://github.com/InstantX-research/InstantX-Flux-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) (Attention Wrapper in this style)
- https://huggingface.co/jimmycarter/LibreFLUX
- https://github.com/bghira/SimpleTuner

Use at your own risk!

## Setup

### Environment

Create a conda environment:
```
conda create -n python=3.11
conda activate
```

### Installation

- Clone the repository
- Change the current directory to `LibreFLUX-IP-Adapter/`
- Install the required dependencies using the `requirements.txt` file
```
git clone https://github.com/YourUsername/LibreFLUX-IP-Adapter/
cd LibreFLUX-IP-Adapter/
pip install -r requirements.txt
```

### Dataset

Place your images in a root directory and reference them in the JSON file.

Your JSON file should be in the following format:
```
[{"image_file": "image1.jpg", "text": "A description of the image"},
{"image_file": "image2.jpg", "text": "Another description"} ]
```

I've included a mini dataset example ( used for validation ) in the `test_dataset` directory


## Training

Example training command:
```
python train_libre_flux.py
--pretrained_model_name_or_path="jimmycarter/LibreFLUX"
--image_encoder_path="google/siglip-so400m-patch14-384"
--data_json_file="laion2b-squareish-1024px_subset/data.json"
--data_root_path="laion2b-squareish-1024px_subset"
--val_data_json_file="test_dataset/data.json"
--val_data_root_path="test_dataset"
--mixed_precision="bf16"
--resolution=512
--train_batch_size=4
--dataloader_num_workers=8
--learning_rate=1e-05
--weight_decay=0.01
--quantize
--output_dir="./output/libreflux_ip_adapter"
--save_steps=1000
--val_steps=500
```
### Key Arguments

- `--pretrained_model_name_or_path`: Base model to use ( This is customized for LibreFlux and/or variants of it )
- `--image_encoder_path`: Image encoder for IP-Adapter ( Hard coded for `google/siglip-so400m-patch14-384` )
- `--data_json_file`: Path to your training data JSON
- `--data_root_path`: Root directory containing training images
- `--val_data_json_file`: Path to validation data JSON
- `--val_data_root_path`: Root directory containing validation images
- `--resolution`: Training resolution
- `--train_batch_size`: Batch size per GPU
- `--quantize`: Enable int8 quantization to save memory
- `--save_steps`: Save checkpoint every N steps
- `--val_steps`: Run validation every N steps

### Memory Optimization

To train on limited GPU memory:

- `--quantize`: Quantize models to int8 (except IP-Adapter being trained)
- `--mixed_precision bf16`: Use bfloat16 precision
- `--use_8bit_adam`: Use 8-bit ADAM optimizer
- `--gradient_checkpointing`: Trade compute for memory

### Resuming Training

To resume from a checkpoint:
- `--pretrained_ip_adapter_path="./output/checkpoint-3000.pt"`

## Inference

**Coming soon!**
