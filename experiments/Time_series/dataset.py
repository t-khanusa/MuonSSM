import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import pandas as pd
import torch
# import xlrd

    
class AFOSR_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = np.array(target, dtype=np.int64)

    def __getitem__(self, index):
        # Read and process data
        data = pd.read_csv(self.data_path[index], header=None).values
        idx = np.linspace(0, len(data[:,0]) - 1, 512, dtype=int)
        samples = data[:, 1:][idx]
        samples = samples.transpose(1,0)
        
        # Add data normalization
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        
        # Convert to tensor
        samples = torch.FloatTensor(samples)
        label = int(self.target[index])
        
            
        return samples, label

    def __len__(self):
        return len(self.data_path)


class UESTC_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target
    

    def __getitem__(self, index):

        y = self.target[index]
        y = np.array(y, int)
        y = torch.from_numpy(y)

        samples = pd.read_csv(self.data_path[index], header=None).values

        a, b, c, d, e, f = np.array(samples[:,0], float)/16384, np.array(samples[:,1], float)/16384, np.array(samples[:,2], float)/16384, np.array(samples[:,3], float)/16.4, np.array(samples[:,4], float)/16.4, np.array(samples[:,5], float)/16.4
        idx = np.linspace(0, len(samples[:,0]) - 1, 512, dtype=int)
        ax, bx, cx, dx, ex, fx = a[idx], b[idx], c[idx], d[idx], e[idx], f[idx]
        samples = np.dstack((ax, bx, cx, dx, ex, fx))[0]

        # Add data normalization
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        samples = samples.transpose(1, 0)

        samples = torch.from_numpy(samples).float()
        

        return samples, y

    def __len__(self):
        return len(self.data_path)
    

class MMAct_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target

    def __getitem__(self, index):
        samples = pd.read_csv(self.data_path[index]).values
        idx = np.linspace(0, len(samples[:,0]) - 1, 512, dtype=int)
        ax, bx, cx, dx, ex, fx, gx, hx, ix, jx, kx, lx = samples[idx, 0], samples[idx, 1], samples[idx, 2], samples[idx, 3], samples[idx, 4], samples[idx, 5], samples[idx, 6], samples[idx, 7], samples[idx, 8], samples[idx, 9], samples[idx, 10], samples[idx, 11]
        samples = np.dstack((ax, bx, cx, dx, ex, fx, gx, hx, ix, jx, kx, lx))[0]
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        samples = samples.transpose(1,0)    
        samples = torch.from_numpy(samples).float()
        label = int(self.target[index])
        return samples, label
    
    def __len__(self):
        return len(self.data_path)
    