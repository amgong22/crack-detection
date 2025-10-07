import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp

# --- CONFIG ---
model_path = r"D:\Thesis Datasets\Project\Masks\best_unet_model.pth"
input_folder = r"D:\Thesis Datasets\Project\Test_Preprocessed"
output_folder = r"D:\Thesis Datasets\Project\Test_Predictions"
mask_output_folder = os.path.join(output_folder, "Prediction Masks")
overlay_output_folder = os.path.join(output_folder, "Overlay")

os.makedirs(mask_output_folder, exist_ok=True)
os.makedirs(overlay_output_folder, exist_ok=True)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load U-Net Model ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Logging Results ---
results = []

# --- Predict Each Image ---
for filename in tqdm(sorted(os.listdir(input_folder)), desc="Predicting"):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print(f"⚠️ Could not read: {filename}")
        continue

    h, w = gray_img.shape
    input_tensor = transform(gray_img).unsqueeze(0).to(device)  # [1, 1, H, W]

    # --- Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # --- Crack Detection ---
    has_crack = np.any(binary_mask > 0)

    # --- Bounding Boxes & Labels ---
    overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if has_crack:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                x, y, bw, bh = cv2.boundingRect(cnt)
                conf = pred_mask[y:y+bh, x:x+bw].mean()  # Confidence = avg mask intensity

                # Draw bounding box + label background
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.rectangle(overlay, (x, y - 20), (x + 70, y), (0, 0, 255), -1)
                cv2.putText(overlay, f"crack {conf:.2f}", (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Save Outputs ---
    mask_out = os.path.join(mask_output_folder, filename)
    overlay_out = os.path.join(overlay_output_folder, filename)
    combined_out = os.path.join(output_folder, "Combined_" + filename)

    cv2.imwrite(mask_out, binary_mask)
    cv2.imwrite(overlay_out, overlay)

    # Optional: side-by-side view (grayscale + mask)
    combined = np.hstack((gray_img, binary_mask))
    cv2.imwrite(combined_out, combined)

    # --- Log Results ---
    results.append({
        "filename": filename,
        "crack_detected": "yes" if has_crack else "no"
    })

# --- Save CSV Report ---
df = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "crack_report.csv")
df.to_csv(csv_path, index=False)

print(f"\n Prediction complete.")
print(f" Overlays saved to: {overlay_output_folder}")
print(f" Masks saved to: {mask_output_folder}")
print(f" Report saved to: {csv_path}")
