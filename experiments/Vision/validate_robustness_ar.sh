export HF_DATASETS_CACHE="./MambaVision/datasets/imagenet-1k"

python validate_imagenet_ar.py \
    --model mamba_vision_T \
    --checkpoint ./output/train/MuonMambaVision_experiment/20260101-234822-mamba_vision_T-224/model_best.pth.tar \
    --imagenet-r-path ./datasets/imagenet-r \
    --batch-size 128