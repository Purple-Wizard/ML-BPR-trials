from keras.models import load_model
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf

def load_images(path, num_images):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]
    

    original_images = [
        np.array(Image.open(file))
        for file in tqdm(image_files, desc="Loading original images", unit="images")
    ]

    images = resize_images(original_images, (64, 64))

    return np.array(images)

def resize_images(images, img_size):
    resized_imgs = []

    for img in tqdm(images, desc=f"Resizing images to {img_size[0]}x{img_size[1]}", unit="images"):
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        resized_imgs.append(img)

    resized_imgs = np.array(resized_imgs)
    resized_imgs = resized_imgs.astype("float32") / 255.0

    return resized_imgs

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

def preprocess_image(image):
    log = apply_log_filter(image, kernel_size=5, sigma=0.5)
    sobel = apply_sobel_filter(image, kernel_size=3)
    image = image.astype(np.float32)
    log = log.astype(np.float32)
    log_color = cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 0.6, log_color, 0.2, 0)
    image = cv2.addWeighted(image, 0.8, cv2.cvtColor(sobel.astype(np.float32), cv2.COLOR_GRAY2BGR), 0.2, 0)
    return image

def load_and_predict_images(image_directory, model_path):
    # Load the images
    images = load_images(image_directory, 1000)
    
    # Load the model
    model = load_model(model_path)

    # List to store the original and predicted images
    original = []
    predicted = []

    # Predict each image one by one
    for image in images:
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        prediction = model.predict(processed_image)
        
        # Store the original and predicted image
        original.append(image)
        predicted.append(prediction)
    
    return original, predicted

def visualize_images(image_pairs):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    # Choose 5 random indices for image display
    indices = np.random.choice(len(image_pairs), size=5, replace=False)

    for i, idx in enumerate(indices):
        original, predicted = image_pairs[idx]

        # compute the column index for the subplot
        col = i % 5

        # Original images in the first row
        axs[0, col].imshow(np.squeeze(original))
        axs[0, col].axis('off')

        # Predicted images in the second row
        axs[1, col].imshow(np.squeeze(predicted))
        axs[1, col].axis('off')

    plt.show()

predictions = load_and_predict_images('dataset_sorted', 'models/best_model.h5')
#visualize_images(predictions)