import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

dataset_path = "archive"
num_images = 100
img_size = (64, 64)

def load_images(path, num_images):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]
    
    original_images = [
        np.array(Image.open(file))
        for file in tqdm(image_files, desc="Loading original images", unit="images")
    ]

    resized = resize_images(original_images, (64, 64))

    processed_images = preprocess_images(resized)

    original_train, original_val, original_test = split_data(resized)
    processed_train, processed_val, processed_test = split_data(processed_images)

    return np.array(original_train), np.array(original_val), np.array(original_test), np.array(processed_train), np.array(processed_val), np.array(processed_test)

def resize_images(images, img_size):
    resized_imgs = []

    for img in tqdm(images, desc=f"Resizing images to {img_size[0]}x{img_size[1]}", unit="images"):
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        resized_imgs.append(img)

    resized_imgs = np.array(resized_imgs)
    resized_imgs = resized_imgs.astype("float32") / 255.0

    return resized_imgs

def display_images(images_lists, titles=["Original", "Gray", "Gaussian", "Canny", "Normalised"]):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    print(len(images_lists))
    for i in range(len(images_lists)):
        if (i > 3):
            break

        for j in range(len(images_lists[0])):
            img = images_lists[i][j]
            axs[i, j].imshow(img, cmap='jet')
            axs[i, j].set_title(titles[j])
    plt.tight_layout()
    plt.show()

def apply_log_filter(img, kernel_size=5, sigma=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    log = cv2.Laplacian(log, cv2.CV_16S, ksize=kernel_size)
    log = cv2.convertScaleAbs(log)
    return log

def apply_sobel_filter(img, kernel_size=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=kernel_size)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobel

def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val)

def apply_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    height, width, channel = image.shape

    # Add salt noise
    num_salt = int(height * width * salt_prob)
    salt_coords = [random.randint(0, height - 1) for _ in range(num_salt)]
    salt_indices = np.unravel_index(salt_coords, (height, width))
    noisy_image[salt_indices] = 1.0

    # Add pepper noise
    num_pepper = int(height * width * pepper_prob)
    pepper_coords = [random.randint(0, height - 1) for _ in range(num_pepper)]
    pepper_indices = np.unravel_index(pepper_coords, (height, width))
    noisy_image[pepper_indices] = 0.0

    return noisy_image

def split_data(images):
    print("Splitting data to train, val, and test...\n_________________________________________________________________")
    images = np.array(images)
    x_train, x_temp = train_test_split(images, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.7, random_state=42)
    return x_train, x_val, x_test

def preprocess_images(images):
    processed_images = []
    display_images_list = []
    titles = ["Original", "S&P", "Gaussian", "Normalised"]

    for i, img in enumerate(tqdm(images, desc="Preprocessing images", unit="images")):
        # Removed the grayscale conversion
        
        salt_pepper = apply_salt_pepper_noise(img.copy(), 0.01, 0.01)  # Apply noise on the original image
        # Reduce noise with Gaussian blur
        blur_vis = cv2.GaussianBlur(salt_pepper, (3, 3), 0)
        
        # Convert to absolute scale is not necessary with 3-channel image
        
        # Normalize the image
        normalized = cv2.normalize(blur_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        display_images_list.append([img, salt_pepper, blur_vis, normalized])
        
        processed_images.append(normalized)

    # display_images(display_images_list, titles)

    return np.array(processed_images, dtype=np.float32)

# load_images(dataset_path, num_images)