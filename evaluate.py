import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score

# --- Paths ---
gt_dir = r"D:\Thesis Datasets\Project\test_dataset_groundtruth\Test_evaluation"
pred_dir = r"D:\Thesis Datasets\Project\Test_Predictions\Prediction Masks"

def binarize_mask(mask):
    return (mask > 127).astype(np.uint8).flatten()

# --- Gather Files ---
gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg'))])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg'))])

matched = list(set(gt_files) & set(pred_files))
if not matched:
    raise ValueError(" No matching files found between ground truth and prediction folders.")

ious, dices, precs, recs, accs = [], [], [], [], []
results = []

for fname in matched:
    gt_path = os.path.join(gt_dir, fname)
    pred_path = os.path.join(pred_dir, fname)

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None or pred_mask is None:
        continue

    if gt_mask.shape != pred_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    gt_bin = binarize_mask(gt_mask)
    pred_bin = binarize_mask(pred_mask)

    iou = jaccard_score(gt_bin, pred_bin, average='binary', zero_division=1)
    dice = f1_score(gt_bin, pred_bin, average='binary', zero_division=1)
    prec = precision_score(gt_bin, pred_bin, zero_division=1)
    rec = recall_score(gt_bin, pred_bin, zero_division=1)
    acc = accuracy_score(gt_bin, pred_bin)

    ious.append(iou); dices.append(dice); precs.append(prec); recs.append(rec); accs.append(acc)
    results.append({"filename": fname, "IoU": iou, "Dice": dice, "Precision": prec, "Recall": rec, "Accuracy": acc})

# --- Results ---
df = pd.DataFrame(results)
df.to_csv(os.path.join(pred_dir, "evaluation_results.csv"), index=False)

print("Evaluation")
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall: {np.mean(recs):.4f}")
print(f"Mean Accuracy: {np.mean(accs):.4f}")
