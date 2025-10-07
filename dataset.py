import os
import cv2
import torch
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        # Only keep image files that have a matching mask (no "_mask" suffix)
        self.img_list = [
            f for f in os.listdir(img_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # same name, no suffix

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Missing file: {img_path} or {mask_path}")

        image = cv2.resize(image, self.size) / 255.0
        mask = (cv2.resize(mask, self.size) / 255.0 > 0.5).astype("float32")

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask
