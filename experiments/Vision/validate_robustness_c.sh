#!/bin/bash
# Validate MambaVision on ImageNet-C (Corruption Robustness)

export HF_DATASETS_CACHE="./MambaVision/datasets/imagenet-1k"

checkpoint='./output/train/ori_MambaVision/20260115-143242-mamba_vision_T-224/model_best.pth.tar'

python validate_imagenet_c.py \
    --model mamba_vision_T \
    --checkpoint=$checkpoint \
    --imagenet-c-path ./datasets/imagenet-c \
    --batch-size 128 \
    --num-workers 4
