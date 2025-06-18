import cv2
import numpy as np
import os

# Input and output folders
input_folder = "calib_images"
output_folder = "calib_data"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exist

# Counter to limit to first 100 images
i = 0
max_images = 100

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    if i >= max_images:
        break  # Stop after 100 images

    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)

        # Step 1: Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping {filename} (could not load)")
            continue

        # Step 2: Resize to model input size (e.g., 640x640 for YOLO)
        img = cv2.resize(img, (640, 640))

        # Step 3: Convert BGR → RGB
        img = img[:, :, ::-1]

        # Step 4: Normalize pixel values to 0.0–1.0
        img = img.astype(np.float32) / 255.0

        # Step 5: Rearrange shape to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # Step 6: Add batch dimension → (1, 3, 640, 640)
        img = np.expand_dims(img, axis=0)

        # Step 7: Save to .npy file (keep same base name)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}.npy")
        np.save(output_path, img)
        i += 1

        print(f"Processed: {filename} → {output_path}")

