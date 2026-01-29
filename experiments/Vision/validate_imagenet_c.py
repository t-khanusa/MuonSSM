#!/usr/bin/env python3
"""
ImageNet-C Corruption Robustness Evaluation Script for MambaVision

Evaluates model performance on ImageNet-C dataset with top-1 accuracy and mCE.
ImageNet-C contains 15 corruption types, each with 5 severity levels.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import numpy as np
from tqdm import tqdm

# Import MambaVision model and timm utilities
from models.mamba_vision import *
from timm.models import create_model
from timm.utils import accuracy, AverageMeter


# AlexNet error rates for mCE normalization (from original ImageNet-C paper)
# Reference: https://github.com/hendrycks/robustness
ALEXNET_ERR = {
    'gaussian_noise': 0.886428,
    'shot_noise': 0.894468,
    'impulse_noise': 0.922640,
    'defocus_blur': 0.819880,
    'glass_blur': 0.826268,
    'motion_blur': 0.785948,
    'zoom_blur': 0.798360,
    'snow': 0.866816,
    'frost': 0.826572,
    'fog': 0.819324,
    'brightness': 0.564592,
    'contrast': 0.853204,
    'elastic_transform': 0.646056,
    'pixelate': 0.717840,
    'jpeg_compression': 0.606500,
    # Extra corruptions (if available)
    'speckle_noise': 0.845388,
    'gaussian_blur': 0.787108,
    'spatter': 0.717512,
    'saturate': 0.658248,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate MambaVision on ImageNet-C',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='mamba_vision_T',
                        help='MambaVision model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--imagenet-c-path', type=str, 
                        default='/home/takas/Templates/MambaVision/object_detection/datasets/imagenet-c',
                        help='Path to ImageNet-C dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Specific corruptions to evaluate (default: all)')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Severity levels to evaluate')
    return parser.parse_args()


def load_model(args, device):
    """Load MambaVision model with checkpoint"""
    print(f'Creating model: {args.model}')
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
    )
    
    # Load checkpoint
    print(f'Loading checkpoint from: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Strip "model." prefix if present
    if any(key.startswith('model.') for key in state_dict.keys()):
        print("Stripping 'model.' prefix from checkpoint keys...")
        state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f'Warning: Missing keys: {len(missing_keys)} keys')
    if unexpected_keys:
        print(f'Warning: Unexpected keys: {len(unexpected_keys)} keys')
    print('Checkpoint loaded successfully.')
    
    model = model.to(device)
    model.eval()
    
    param_count = sum([m.numel() for m in model.parameters()])
    print(f'Model created, param count: {param_count:,}')
    
    return model


def evaluate_corruption(model, loader, device):
    """Evaluate model on a single corruption/severity and return top-1 accuracy"""
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc='Evaluating', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
    
    return top1.avg, top5.avg


def get_corruption_list(imagenet_c_path):
    """Get list of available corruption types"""
    # Standard 15 corruptions
    standard_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
    
    # Extra corruptions (may or may not exist)
    extra_corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    
    # Check which corruptions actually exist
    available = []
    for corruption in standard_corruptions + extra_corruptions:
        if os.path.exists(os.path.join(imagenet_c_path, corruption)):
            available.append(corruption)
    
    return available


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    cudnn.benchmark = True
    
    # Load model
    model = load_model(args, device)
    
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Transform for ImageNet-C (images are already distorted, just normalize)
    test_transform = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean, std)
    ])
    
    # Get corruption types
    if args.corruptions:
        corruptions = args.corruptions
    else:
        corruptions = get_corruption_list(args.imagenet_c_path)
    
    print(f'\nEvaluating on {len(corruptions)} corruption types')
    print(f'Severity levels: {args.severities}')
    print('='*80)
    
    # Store results
    results = {}
    error_rates = {}
    
    for corruption in corruptions:
        print(f'\n>>> Corruption: {corruption}')
        corruption_errors = []
        corruption_accs = []
        
        for severity in args.severities:
            corruption_path = os.path.join(args.imagenet_c_path, corruption, str(severity))
            
            if not os.path.exists(corruption_path):
                print(f'    Severity {severity}: Path not found - {corruption_path}')
                continue
            
            # Load dataset
            dataset = dset.ImageFolder(root=corruption_path, transform=test_transform)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Evaluate
            top1_acc, top5_acc = evaluate_corruption(model, loader, device)
            error_rate = (100 - top1_acc) / 100.0  # Convert to error rate [0, 1]
            
            corruption_errors.append(error_rate)
            corruption_accs.append(top1_acc)
            
            print(f'    Severity {severity}: Top-1 Acc: {top1_acc:.2f}%, Error: {error_rate*100:.2f}%')
        
        if corruption_errors:
            # Average error across severities
            avg_error = np.mean(corruption_errors)
            avg_acc = np.mean(corruption_accs)
            
            error_rates[corruption] = avg_error
            results[corruption] = {
                'severity_errors': corruption_errors,
                'severity_accs': corruption_accs,
                'avg_error': avg_error,
                'avg_acc': avg_acc,
            }
            
            print(f'    Average - Top-1 Acc: {avg_acc:.2f}%, Error: {avg_error*100:.2f}%')
    
    # Calculate mCE (mean Corruption Error)
    print('\n' + '='*80)
    print('Results Summary')
    print('='*80)
    
    ce_values = []
    ce_dict = {}
    
    print(f'\n{"Corruption":<25} {"Avg Acc (%)":<12} {"Avg Err (%)":<12} {"CE (%)":<10}')
    print('-'*60)
    
    for corruption in corruptions:
        if corruption not in results:
            continue
        
        avg_error = results[corruption]['avg_error']
        avg_acc = results[corruption]['avg_acc']
        
        # Calculate CE normalized by AlexNet
        if corruption in ALEXNET_ERR:
            ce = avg_error / ALEXNET_ERR[corruption]
            ce_values.append(ce)
            ce_dict[corruption] = ce
            print(f'{corruption:<25} {avg_acc:<12.2f} {avg_error*100:<12.2f} {ce*100:<10.2f}')
        else:
            print(f'{corruption:<25} {avg_acc:<12.2f} {avg_error*100:<12.2f} {"N/A":<10}')
    
    print('-'*60)
    
    # Calculate mCE
    if ce_values:
        mce = np.mean(ce_values) * 100
        print(f'\nmCE (mean Corruption Error): {mce:.2f}%')
    
    # Calculate mean accuracy
    all_accs = [results[c]['avg_acc'] for c in corruptions if c in results]
    if all_accs:
        mean_acc = np.mean(all_accs)
        print(f'Mean Accuracy across corruptions: {mean_acc:.2f}%')
    
    # Print by category
    print('\n' + '='*80)
    print('Results by Category')
    print('='*80)
    
    categories = {
        'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
        'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur'],
        'Weather': ['snow', 'frost', 'fog', 'brightness', 'spatter'],
        'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate'],
    }
    
    for category, corruption_list in categories.items():
        category_ces = [ce_dict[c] for c in corruption_list if c in ce_dict]
        category_accs = [results[c]['avg_acc'] for c in corruption_list if c in results]
        
        if category_ces:
            print(f'{category}: mCE = {np.mean(category_ces)*100:.2f}%, Mean Acc = {np.mean(category_accs):.2f}%')
    
    # Save detailed results to file
    results_file = os.path.join(os.path.dirname(args.checkpoint), 'imagenet_c_results.txt')
    try:
        with open(results_file, 'w') as f:
            f.write('ImageNet-C Evaluation Results\n')
            f.write('='*60 + '\n')
            f.write(f'Model: {args.model}\n')
            f.write(f'Checkpoint: {args.checkpoint}\n')
            f.write(f'mCE: {mce:.2f}%\n' if ce_values else '')
            f.write(f'Mean Accuracy: {mean_acc:.2f}%\n\n' if all_accs else '\n')
            
            f.write('Detailed Results:\n')
            f.write('-'*60 + '\n')
            for corruption in corruptions:
                if corruption not in results:
                    continue
                r = results[corruption]
                f.write(f'{corruption}:\n')
                for i, (err, acc) in enumerate(zip(r['severity_errors'], r['severity_accs']), 1):
                    f.write(f'  Severity {i}: Acc={acc:.2f}%, Err={err*100:.2f}%\n')
                f.write(f'  Average: Acc={r["avg_acc"]:.2f}%, Err={r["avg_error"]*100:.2f}%')
                if corruption in ce_dict:
                    f.write(f', CE={ce_dict[corruption]*100:.2f}%')
                f.write('\n\n')
        print(f'\nResults saved to: {results_file}')
    except Exception as e:
        print(f'Warning: Could not save results file: {e}')


if __name__ == '__main__':
    main()
