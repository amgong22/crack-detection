import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

# --- Dataset ---
class CrackSegmentationDataset(Dataset):
    def __init__(self, input_dir, mask_dir, augment=False):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.augment = augment

        input_files = set(os.listdir(input_dir))
        mask_files = set(os.listdir(mask_dir))
        self.filenames = sorted(list(input_files & mask_files))
        print(f"Total matched samples: {len(self.filenames)}")

        self.transform = T.Compose([
            T.ToTensor(),
        ])

        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = cv2.imread(os.path.join(self.input_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Unreadable file: {fname}")

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.augment:
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            # Randomly apply augmentation
            if torch.rand(1) < 0.5:
                stacked = torch.cat((img, mask), dim=0)
                stacked = self.augmentations(stacked)
                img, mask = stacked[0].unsqueeze(0), stacked[1].unsqueeze(0)
            return img, mask

        return torch.tensor(img), torch.tensor(mask)


# --- IoU & Dice ---
def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = (y_pred.cpu().numpy().flatten() > 0.5).astype(np.uint8)
    iou = jaccard_score(y_true, y_pred, zero_division=1)
    dice = f1_score(y_true, y_pred, zero_division=1)
    return iou, dice


# --- Training ---
def main():
    input_dir = r"D:\Thesis Datasets\Project\Images"
    mask_dir = r"D:\Thesis Datasets\Project\Masks"

    dataset = CrackSegmentationDataset(input_dir, mask_dir, augment=True)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses, val_losses, val_ious, val_dices = [], [], [], []
    best_iou = 0.0

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss, val_iou, val_dice = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()

                for i in range(x.size(0)):
                    iou, dice = compute_metrics(y[i], torch.sigmoid(pred[i]))
                    val_iou += iou
                    val_dice += dice

        val_loss /= len(val_loader)
        val_iou /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(mask_dir, "best_unet_model.pth"))

    # --- Plots ---
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(mask_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(val_ious, label='Validation IoU')
    plt.plot(val_dices, label='Validation Dice')
    plt.legend()
    plt.title("Validation IoU and Dice over Epochs")
    plt.savefig(os.path.join(mask_dir, "val_metrics.png"))
    plt.close()

    print(" Training complete. Best model saved.")

if __name__ == "__main__":
    main()
