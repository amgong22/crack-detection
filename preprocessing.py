import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
input_dir = r"D:\Thesis Datasets\Project\Kaggle Dataset"
output_filtered_dir = r"D:\Thesis Datasets\Project\kaggle_filtered"
output_edges_dir = r"D:\Thesis Datasets\Project\kaggle_edges"
# Create output folders
os.makedirs(output_filtered_dir, exist_ok=True)
os.makedirs(output_edges_dir, exist_ok=True)

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Process all images
for fname in tqdm(os.listdir(input_dir), desc="Preprocessing images"):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(input_dir, fname)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue

    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)

    # Step 2: Bilateral filter
    filtered = cv2.bilateralFilter(image_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Sobel
    sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_abs = cv2.convertScaleAbs(sobel_mag)

    # Step 4: Otsu
    _, otsu_mask = cv2.threshold(sobel_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Morph closing
    cleaned = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    # Save
    cv2.imwrite(os.path.join(output_filtered_dir, fname), filtered)
    cv2.imwrite(os.path.join(output_edges_dir, fname), cleaned)

print("Preprocessing complete.")
