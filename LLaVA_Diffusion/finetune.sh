#!/bin/bash -l

# Run DeepSpeed without manually specifying --include
deepspeed path/to/train_script.py \
    --deepspeed path/to/deepspeed_config.json \
    --model_name_or_path model_path \
    --version mpt \
    --data_path /path/to/data/chat.json \
    --image_folder /path/to/images \
    --vision_tower vision_model/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /path/to/pretrained/mm_projector.bin \
    --pretrain_diffusion_mm_mlp_adapter /path/to/pretrained/diffusion_mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /path/to/output_directory \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mpt_attn_impl "torch" \
