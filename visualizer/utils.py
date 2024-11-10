"""
Utility functions for the visualizer application which includes image loading, processing, and saving.

"""

import cv2
import numpy as np
import torch
import os
import csv

def preprocess_image(image, img_size):
    """Resizes and normalizes the image for model input."""
    image = cv2.resize(image, (img_size, img_size))
    image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0
    return image

def postprocess_image(tensor):
    """Converts a tensor back into an image."""
    img = tensor.squeeze(0).detach().numpy().transpose((1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def normalize_to_uint8(img):
    """Normalize a given image (either residual or regular image) to uint8 [0, 255]."""
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def generate_heatmap(residuals, original_img, blend_factor):
    """Generates a heatmap based on residuals and blends it with the original image."""
    normalized_residuals = normalize_to_uint8(residuals)
    # Contrast Limited Adaptive Histogram Equalization(CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4)) # Adjust clipLimit for more contrast
    enhanced_residuals = clahe.apply(normalized_residuals)
    heatmap = cv2.applyColorMap(enhanced_residuals, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1 - blend_factor, heatmap, blend_factor, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended_rgb

def load_images(folder_path, visualizer):
    """ A helper function to load images from a folder. """
    visualizer.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not visualizer.image_files:
        print(f"No images found in {folder_path}!")
        return

    visualizer.current_image_idx = 0
    visualizer.image_selector.clear()
    visualizer.image_selector.addItems(visualizer.image_files)
    load_image_at_index(visualizer, visualizer.current_image_idx)

def load_image_at_index(visualizer, index):
    """Loads a specific image at the given index for the visualizer."""
    if visualizer.image_files:
        folder_path = os.path.join(os.getcwd(), "anomaly-detection", "Bowtie", "test", visualizer.current_folder)
        image_path = os.path.join(folder_path, visualizer.image_files[index])
        visualizer.original_img = cv2.imread(image_path)
        
        if visualizer.original_img is None:
            print(f"Failed to load image from {image_path}")
            return
        
        visualizer.original_img = cv2.cvtColor(visualizer.original_img, cv2.COLOR_BGR2RGB)
        visualizer.original_img = cv2.resize(visualizer.original_img, (256, 256))
        visualizer.original_img_tensor = preprocess_image(visualizer.original_img, visualizer.img_size)

        # Perform anomaly detection
        with torch.no_grad():
            visualizer.reconstructed_img_tensor = visualizer.cae_model(visualizer.original_img_tensor.to(visualizer.device)).cpu()
        
        visualizer.reconstructed_img = postprocess_image(visualizer.reconstructed_img_tensor)

        # Apply anomaly detection logic
        threshold = visualizer.threshold_slider.value()
        original_img_float = visualizer.original_img.astype(np.float32)
        reconstructed_img_float = visualizer.reconstructed_img.astype(np.float32)
        residuals = np.abs(original_img_float - reconstructed_img_float)
        smoothed_residuals = cv2.GaussianBlur(residuals, (3, 3), 0)
        residuals_gray = np.mean(smoothed_residuals, axis=2)
        initial_anomaly_mask = residuals_gray > threshold

        # Morphological operations and connected component analysis
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        processed_mask = cv2.morphologyEx(initial_anomaly_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)

        valid_mask = np.zeros(processed_mask.shape, dtype=bool)
        min_size = 10
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_size:
                valid_mask[labels == label] = True

        anomaly_count = np.sum(valid_mask)
        visualizer.anomaly_count_label.setText(f"Anomalies Detected: {anomaly_count}")

        # Update true and predicted labels based on anomalies detected
        visualizer.true_labels.append(0 if visualizer.current_folder == 'good' else 1)
        visualizer.predicted_labels.append(1 if anomaly_count > 0 else 0)

        # Update displayed images
        visualizer.update_images()

def save_anomaly_counts_to_csv(visualizer, label):
    """
    Saves the anomaly counts for each image in visualizer.anomaly_counts to a CSV file in the root directory as "anomaly.csv".
    :param label: The true label for each image in visualizer.anomaly_counts (0 for good, 1 for reject).
    """
    root_dir = os.path.abspath(os.getcwd())
    csv_path = os.path.join(root_dir, "anomaly.csv")

    # Open the file in append mode with newline='' to add new entries without overwriting
    file_mode = 'a' if os.path.exists(csv_path) else 'w'
    with open(csv_path, mode=file_mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header only if the file is new
        if file_mode == 'w':
            writer.writerow(["Image", "AnomaliesDetected", "TrueLabel"])

        # Write each image, its anomaly count, and the label
        for image_path, anomaly_count in visualizer.anomaly_counts.items():
            writer.writerow([image_path, anomaly_count, label])

    print(f"Anomaly counts saved to {csv_path}")

def process_all_images_and_save(visualizer, label):
    """
    Process all images in the current folder and save anomaly counts for each to a CSV file.
    :param label: The true label for all images in the current folder (0 for good, 1 for reject).
    """
    if not visualizer.image_files:
        print("No images found to process.")
        return

    # Clear the dictionary to avoid mixing previous results
    visualizer.anomaly_counts.clear()

    # Process each image in the folder and store the anomaly count
    for idx, image_path in enumerate(visualizer.image_files):
        visualizer.current_image_idx = idx
        load_image_at_index(visualizer, idx)  # Pass visualizer as the first argument

    # After processing all images, save the results with the specified label
    save_anomaly_counts_to_csv(visualizer, label)
    print(f"All images processed and saved to anomaly.csv with label {label}")
