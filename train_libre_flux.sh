#!/bin/bash

# This script runs the LibreFLUX training process

/venv/main/bin/python train_libre_flux.py \
  --pretrained_model_name_or_path="jimmycarter/LibreFLUX" \
  --image_encoder_path="google/siglip-so400m-patch14-384" \
  --data_json_file="laion2b-squareish-1024px/data.json" \
  --data_root_path="laion2b-squareish-1024px" \
  --val_data_json_file="test_dataset/data.json" \
  --val_data_root_path="test_dataset" \
  --mixed_precision="bf16" \
  --resolution=512 \
  --train_batch_size=4 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --quantize \
  --output_dir="output" \
  --save_steps=1000 \
  --val_steps=500 \

echo "Training script finished."