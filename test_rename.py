import os
import shutil

# Input and output directories
source_dir = r"D:\Thesis Datasets\Project\hehe"
output_dir = r"D:\Thesis Datasets\Project\Test Dataset"

os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
image_exts = ('.png', '.jpg', '.jpeg')

# Copy and rename only images (exclude masks)
counter = 1
for fname in sorted(os.listdir(source_dir)):
    if "_mask" in fname.lower() or not fname.lower().endswith(image_exts):
        continue

    src_path = os.path.join(source_dir, fname)
    dst_path = os.path.join(output_dir, f"{counter:03}.png")
    shutil.copy(src_path, dst_path)
    counter += 1

print(f"✅ Copied {counter - 1} images to: {output_dir}")
