#!/bin/bash -l

python train.py \
    --dataset_flag "timm/oxford-iiit-pet" \
    --output_dir "./outputs" \
    --model_name "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --diffusion_timestep 25 \
    --diffusion_layer "up_ft:1" \
    --learning_rate 1e-4 \
    --num_epochs 90 \
    --batch_size 16 \
    --num_classes 37 \
    --prompt_type "empty" \
    --pooling_strategy "GAP" \
    --dropout 0.0 
