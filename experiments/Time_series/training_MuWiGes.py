import os
import glob
import argparse
import torch.utils
# import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from get_model import get_model
from dataset import AFOSR_Dataset
import copy
import time
import math
import csv
import numpy as np
# from generate_LinOSS_pytorch import create_pytorch_model
# from mamba_linoss import create_mamba_linoss_model


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup learning rate scheduler"""
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * lr_scale for _ in self.optimizer.param_groups]
        else:
            return [self.base_lr for _ in self.optimizer.param_groups]




def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, patience=7, verbose=False):
    verbose = True
    """
    Train model with optimizations including mixed precision training
    """
    since = time.time()

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    early_stop = False
    
    # Track training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    


    for epoch in range(num_epochs):
        # if early_stop:
            # if verbose:
            #     print(f"Early stopping triggered at epoch {epoch}")
            # break
            
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs = model(inputs)
                        
                        # Numerical stability check for NaN/Inf
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                print(f"⚠️  NaN/Inf in model outputs! Skipping batch...")
                                continue
                            
                        loss = criterion(outputs, labels)
                        loss.backward()

                        # Check gradients for NaN/Inf
                        has_bad_grads = False
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                if not torch.isfinite(param.grad).all():
                                    print(f"⚠️  NaN/Inf in gradient of {name}!")
                                    has_bad_grads = True
                        
                        if has_bad_grads:
                            print("    Skipping optimizer step due to bad gradients...")
                            continue
                        
                        # Optimizer step (no gradient clipping, no scaler)
                        optimizer.step()
                        
                        # Apply gradient clipping
                        # if use_grad_clipping:
                        #     scaler.unscale_(optimizer)
                        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # scaler.step(optimizer)
                        # scaler.update()
                    else:
                        # Regular forward pass for validation - FIXED deprecated API
                        # with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        
                        # Numerical stability check for NaN/Inf
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"⚠️  NaN/Inf in validation outputs! Skipping batch...")
                            continue
                        
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        early_stop = True
                
                if scheduler is not None:
                    scheduler.step(epoch_acc)

        if verbose:
            print()

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, best_acc.item(), time_elapsed, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set with mixed precision"""
    model.eval()
    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)
    
    test_acc = running_corrects.double() / total_samples
    test_loss = running_loss / total_samples
    return test_acc.item(), test_loss


'''
DATA
'''
data_train, target_train = [], []
data_val, target_val = [],[]

data_fold= "/project/khanhnt/HAR_dataset/AFOSR_data"
# Add debug prints to verify data distribution
class_counts_train = {i: 0 for i in range(12)}  # Assuming 12 classes (G1-G12)
class_counts_val = {i: 0 for i in range(12)}

for data_path in glob.glob(os.path.join(data_fold, "*", "*", "*","*.csv")):
    label = int(data_path.split("/")[-1][:2]) - 1
    if data_path.split("/")[-4] == "data_train":
        data_train.append(data_path)
        target_train.append(label)
        class_counts_train[label] += 1
    else:
        data_val.append(data_path)
        target_val.append(label)
        class_counts_val[label] += 1

batch_size = 16

print("Train: ", len(data_train))
print("Val: ", len(data_val))


    
    # Create class distribution plot

train_dataset = AFOSR_Dataset(data_train, target_train)
val_dataset = AFOSR_Dataset(data_val, target_val)



batch_size = 16
num_workers = 4 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}


'''
MODEL
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--name', action="store", dest='name', default="MMA")
parser.add_argument('--d_model', action="store", dest='d_model', default=128)
parser.add_argument('--d_state', action="store", dest='d_state', default=64)
parser.add_argument('--num_classes', action="store", dest='num_classes', default=12)
parser.add_argument('--input_channels', action="store", dest='input_channels', default=6)
parser.add_argument('--n_layers', action="store", dest='n_layers', default=2)
parser.add_argument('--momentum_beta', action="store", dest='momentum_beta', type=float, default=0)
parser.add_argument('--momentum_alpha', action="store", dest='momentum_alpha', type=float, default=0.6)
args = parser.parse_args()

# Grid search parameters
# Define ranges for beta and alpha
beta_values = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]  # You can modify these values
alpha_values = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]  # You can modify these values

# Results storage
results = []
best_acc = 0.0
best_beta = None
best_alpha = None
best_model_state = None

# Setup CSV file for incremental logging
csv_filename = 'grid_search_results_MuWiGes.csv'
fieldnames = ['momentum_beta', 'momentum_alpha', 'warmup_val_acc', 'val_acc', 
              'test_acc', 'final_val_acc', 'test_loss', 'warmup_time', 
              'train_time', 'total_time']

# Open CSV file and write header
csvfile = open(csv_filename, 'w', newline='')
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
csvfile.flush()  # Ensure header is written immediately

print("=" * 80)
print("Starting Grid Search over momentum_beta and momentum_alpha")
print(f"Beta values: {beta_values}")
print(f"Alpha values: {alpha_values}")
print(f"Total combinations: {len(beta_values) * len(alpha_values)}")
print(f"Results will be logged incrementally to: {csv_filename}")
print("=" * 80)

# Grid search loop
for beta in beta_values:
    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Training with momentum_beta={beta}, momentum_alpha={alpha}")
        print(f"{'='*80}")
        
        # Update args with current beta and alpha
        args.momentum_beta = float(beta)
        args.momentum_alpha = float(alpha)
        
        # Create model with current parameters
        model = get_model(args).to(device)
        
        # Print model info only for first combination
        if beta == beta_values[0] and alpha == alpha_values[0]:
            params_to_update = model.parameters()
            print("Params to learn:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name} will be updated")
                else:
                    print(f"{name} will not be updated")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
        
        # Setup optimizer and scheduler - ENHANCED for better performance
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        # Enhanced optimizer with momentum-aware learning rate
        base_lr = 8e-4 if beta > 0.8 else 1e-3  # Lower LR for high momentum
        optimizer = torch.optim.AdamW(  # AdamW for better regularization
            model.parameters(), 
            lr=base_lr, 
            weight_decay=1e-4,  # Slightly higher weight decay
            betas=(0.9, 0.999)   # Optimized beta values
        )
        
        # Enhanced scheduler with warmup
        warmup_scheduler = WarmupLR(optimizer, warmup_epochs=3, base_lr=base_lr)
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.65, patience=4, min_lr=1e-6
        )
        
        # Training phase
        print("Warmup training (10 epochs)...")
        model, warmup_val_acc, warmup_time, warmup_history = train_model(
            model, dataloaders, criterion, optimizer, warmup_scheduler, 
            num_epochs=10, patience=10, verbose=False
        )
        
        print("Main training (50 epochs)...")
        model, val_acc, train_time, main_history = train_model(
            model, dataloaders, criterion, optimizer, main_scheduler, 
            num_epochs=40, patience=10, verbose=False
        )
        
        # Test evaluation (using validation set as test set)
        test_acc, test_loss = evaluate_model(model, val_loader, criterion, device)
        
        # Final validation accuracy calculation
        running_corrects = 0
        phase = 'val'  # Use validation set for testing
        
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
        # Store results
        result = {
            'momentum_beta': beta,
            'momentum_alpha': alpha,
            'warmup_val_acc': warmup_val_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'final_val_acc': epoch_acc.item(),
            'test_loss': test_loss,
            'warmup_time': warmup_time,
            'train_time': train_time,
            'total_time': warmup_time + train_time
        }
        results.append(result)
        
        # Immediately write result to CSV file
        writer.writerow(result)
        csvfile.flush()  # Ensure data is written to disk immediately
        print(f"  ✓ Result logged to {csv_filename}")
        
        print(f"\nResults for beta={beta}, alpha={alpha}:")
        print(f"  Warmup Val Acc: {warmup_val_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Final Val Acc: {epoch_acc.item():.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Total Time: {(warmup_time + train_time)/60:.2f} minutes")
        
        # Track best model
        if epoch_acc.item() > best_acc:
            best_acc = epoch_acc.item()
            best_beta = beta
            best_alpha = alpha
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  *** NEW BEST ACCURACY: {best_acc:.4f} ***")
        
        # Clear GPU memory
        del model, optimizer, warmup_scheduler, main_scheduler
        torch.cuda.empty_cache()

# Close CSV file
csvfile.close()

# Print summary
print(f"\n{'='*80}")
print(f"All results have been logged incrementally to {csv_filename}")
print(f"{'='*80}")
print(f"\n{'='*80}")
print("GRID SEARCH SUMMARY")
print(f"{'='*80}")
print(f"Best Accuracy: {best_acc:.4f}")
print(f"Best momentum_beta: {best_beta}")
print(f"Best momentum_alpha: {best_alpha}")
print(f"\nAll results saved to {csv_filename}")
print(f"{'='*80}")

# predictions = np.concatenate(predictions)
# labelss = np.concatenate(labelss)

# from sklearn.metrics import confusion_matrix, classification_report
# from matplotlib import pyplot as plt
# import seaborn as sns

# def plot_confusion_matrix(y_test,y_scores, classNames):
#     # y_test=np.argmax(y_test, axis=1)
#     # y_scores=np.argmax(y_scores, axis=1)
#     classes = len(classNames)
#     cm = confusion_matrix(y_test, y_scores)
#     print("**** Confusion Matrix ****")
#     print(cm)
#     print("**** Classification Report ****")
#     print(classification_report(y_test, y_scores, target_names=classNames))
#     con = np.zeros((classes,classes))
#     for x in range(classes):
#         for y in range(classes):
#             con[x,y] = round(cm[x,y]/np.sum(cm[x,:]), 2)

#     plt.figure(figsize=(90,90))
#     sns.set(font_scale=4.5) # for label size
#     df = sns.heatmap(con, annot=True,fmt='.2', xticklabels= classNames , yticklabels= classNames)
#     df.figure.savefig("UESTC_cf_transformer.png")

# plot_confusion_matrix(labelss,predictions, modulation_list)

# wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(predictions, labelss, class_names=modulation_list)})

# wandb.log({"test_acc": epoch_acc})

# wandb.finish()