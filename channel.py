import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class CrackDataset(Dataset):
    def __init__(self, img_dir, edge_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.edge_dir = edge_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img = cv2.imread(os.path.join(self.img_dir, fname), cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(os.path.join(self.edge_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        # Normalize and stack [2, H, W]
        x = np.stack([img, edge], axis=0) / 255.0  # shape: (2, H, W)
        y = np.expand_dims(mask / 255.0, axis=0)   # shape: (1, H, W)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
