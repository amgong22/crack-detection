import os
import shutil

# Source folder containing images and masks
test_folder = r"D:\Thesis Datasets\Project\hehe"

# Output folders
output_base = r"D:\Thesis Datasets\Project\Test Dataset"
output_images_dir = os.path.join(output_base, "Images")
output_masks_dir = os.path.join(output_base, "Masks")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Helper: extract base name for matching (ignore _mask)
def clean_name(filename):
    return os.path.splitext(os.path.basename(filename))[0].lower().replace(" ", "").replace("_mask", "")

# Store matched files
image_files = {}
mask_files = {}

# Scan the folder
for f in os.listdir(test_folder):
    path = os.path.join(test_folder, f)
    if not os.path.isfile(path):
        continue
    if "_mask" in f.lower():
        mask_files[clean_name(f)] = path
    else:
        image_files[clean_name(f)] = path

# Match by cleaned name and copy
counter = 1
for key in sorted(set(image_files) & set(mask_files)):
    filename = f"{counter:03}.png"
    shutil.copy(image_files[key], os.path.join(output_images_dir, filename))
    shutil.copy(mask_files[key], os.path.join(output_masks_dir, filename))
    counter += 1

print(f"✅ Copied {counter-1} matched image-mask pairs to:\n- {output_images_dir}\n- {output_masks_dir}")
