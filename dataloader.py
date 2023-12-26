import torch
import numpy as np

from torch_snippets import *
from joblib import Parallel, delayed
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, items_a, items_b):
        self.items_A = self.get_image(items_a)
        self.items_B = self.get_image(items_b)
        
        
    def __len__(self):
        return min(len(self.items_A), len(self.items_B))
    
    
    def __getitem__(self, index):
        idx_a = np.random.randint(0, len(self.items_A))
        idx_b = np.random.randint(0, len(self.items_B))

        return self.items_A[idx_a], self.items_B[idx_b]
    
    
    def get_image(self, items):
        def read_image(path):
            return Image.open(path)
        
        with Parallel(n_jobs=-1) as parallel:
            result = parallel(delayed(read_image)(path) for path in items)
        
        return result
    

class DataCollator():

    def __call__(self, data):
        imsA, imsB = list(zip(*data))
        
        imsA = np.stack(imsA, axis=0) / 255.0
        imsB = np.stack(imsB, axis=0) / 255.0
        
        return torch.Tensor(imsA).permute(0, 3, 1, 2), torch.Tensor(imsB).permute(0, 3, 1, 2)