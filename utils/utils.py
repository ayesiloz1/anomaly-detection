import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def mkdirs(paths):
    """
    Create directories if they don't exist.
    
    Args:
    - paths: A single path or a list of paths to create.
    """
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    elif not os.path.exists(paths):
        os.makedirs(paths)


def save_images(images, image_paths, data):
    """
    Save images to the specified path.

    Args:
    - images: The image tensor or array to be saved.
    - image_paths: The directory path where the image should be saved.
    - data: A dictionary that includes the label and path info for the image.
    """
    # Convert tensor to numpy array if necessary
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()  # Ensure it's on CPU and detached from the graph
    
    # Convert from NumPy array to PIL image
    images = Image.fromarray(images.astype(np.uint8))
    
    # Get label and file name from data
    label = data["label"][0]
    file_name = data["path"][0].split("/")[-1]  # Extract file name
    
    # Create the directory if it doesn't exist
    if not os.path.exists(image_paths):
        os.makedirs(image_paths)  # Make sure the directory exists
    
    # Construct full image path with label and file name
    full_image_path = os.path.join(image_paths, f"{label}_{file_name}")
    
    # Save the image
    images.save(full_image_path)
    print(f"Image saved at: {full_image_path}")


def convert2img(image, imtype=np.uint8):
    """
    Convert a Tensor or NumPy array to an image suitable for saving.

    Args:
    - image: A PyTorch tensor or NumPy array representing the image.
    - imtype: The desired image type (default is uint8).
    
    Returns:
    - image: The image converted to a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
        assert len(image.squeeze().shape) < 4
    if image.dtype != np.uint8:
        image = (np.transpose(image.squeeze(), (1, 2, 0)) * 0.5 + 0.5) * 255
    return image.astype(imtype)


def plt_show(img):
    """
    Display an image using matplotlib.

    Args:
    - img: The image tensor to be displayed.
    """
    img = torchvision.utils.make_grid(img.cpu().detach())
    img = img.numpy()
    if img.dtype != "uint8":
        img_numpy = img * 0.5 + 0.5
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()


def compare_images(real_img, generated_img, threshold=0.4):
    """
    Compare real and generated images to detect anomalies.

    Args:
    - real_img: The original input image.
    - generated_img: The reconstructed image by the autoencoder.
    - threshold: The anomaly detection threshold (default is 0.4).

    Returns:
    - anomaly_img: The anomaly-detected image as a NumPy array.
    """
    generated_img = generated_img.type_as(real_img)
    diff_img = np.abs(generated_img - real_img)
    real_img = convert2img(real_img)
    generated_img = convert2img(generated_img)
    diff_img = convert2img(diff_img)

    threshold_value = (threshold * 0.5 + 0.5) * 255
    diff_img[diff_img <= threshold_value] = 0

    anomaly_img = np.zeros_like(real_img)
    anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img > 0)[0], np.where(diff_img > 0)[1]] = [200, 0, 0]  # Red highlighting for anomalies

    # Display comparison using matplotlib
    fig, plots = plt.subplots(1, 4)

    fig.set_figwidth(9)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(real_img, label="real")
    plots[1].imshow(generated_img)
    plots[2].imshow(diff_img)
    plots[3].imshow(anomaly_img)

    plots[0].set_title("Real")
    plots[1].set_title("Generated")
    plots[2].set_title("Difference")
    plots[3].set_title("Anomaly Detection")
    plt.show()

    return anomaly_img  # Return the NumPy array for saving
