# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class GeometricShapesDataset(Dataset):
    """
    随机生成简单几何图形（圆、方块）用于二值分割
    """
    def __init__(self, num_samples=200, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        shape_type = np.random.choice(['circle', 'square'])
        x, y = np.random.randint(10, self.image_size-10, size=2)
        size = np.random.randint(5, 15)
        
        if shape_type == 'circle':
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i - x)**2 + (j - y)**2 <= size**2:
                        img[i,j] = 1.0
                        mask[i,j] = 1.0
        else:
            x1, y1 = max(0, x-size), max(0, y-size)
            x2, y2 = min(self.image_size, x+size), min(self.image_size, y+size)
            img[x1:x2, y1:y2] = 1.0
            mask[x1:x2, y1:y2] = 1.0
        
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        return torch.tensor(img), torch.tensor(mask)
