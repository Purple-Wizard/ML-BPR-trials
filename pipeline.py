import numpy as np
from hashing_model import HashingLayer
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from PIL import Image
from tqdm import tqdm
import os
import random
import cv2

def load_images(path, num_images):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]
    
    # original_images = [
    #     np.array(Image.open(file))
    #     for file in tqdm(image_files, desc="Loading original images", unit="images")
    # ]

    original_images = []
    for file in tqdm(image_files, desc="Loading original images", unit="images"):
        img = Image.open(file)
        # Check if image has more than 3 channels (e.g. it's an RGBA image)
        if len(img.split()) > 3:
            # Convert to RGB
            r, g, b, _ = img.split()
            img = Image.merge('RGB', (r, g, b))
        original_images.append(np.array(img))

    resized = resize_images(original_images, (128, 128))

    return tf.data.Dataset.from_tensor_slices((np.array(resized), np.array(resized)))

def resize_images(images, img_size):
    resized_imgs = []

    for img in tqdm(images, desc=f"Resizing images to {img_size[0]}x{img_size[1]}", unit="images"):
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        resized_imgs.append(img)

    resized_imgs = np.array(resized_imgs)
    resized_imgs = resized_imgs.astype("float32") / 255.0

    return resized_imgs

# Load the encoder and decoder models
encoder = load_model('models/encoder.h5', custom_objects={'HashingLayer': HashingLayer})
decoder = load_model('models/decoder.h5')

# model = Model(encoder.input, decoder(encoder.output))

images = load_images('gmaps_test', 10)

encoded_imgs = encoder.predict(images.batch(32))
decoded_imgs = decoder.predict(encoded_imgs)

images_iter = next(iter(images.batch(32)))

n = 10
random_indices = random.sample(range(images_iter[0].shape[0]), n)
plt.figure(figsize=(20, 4))
for i, idx in enumerate(random_indices):

    ssims = [structural_similarity(images_iter[0][idx].numpy().reshape(128, 128, 3), decoded_imgs[idx].reshape(128, 128, 3), data_range=1, multichannel=True, win_size=3) for i in range(len(decoded_imgs))]

    ax = plt.subplot(2, n, i+1)
    plt.imshow(images_iter[0][idx])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if (i < 1):
        og = images_iter[0][idx].numpy()
        # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        og = (og * 255).astype(np.uint8)
        og = Image.fromarray(og)
        decoded = decoded_imgs[idx]
        decoded = (decoded * 255).astype(np.uint8)
        decoded = Image.fromarray(decoded)
        og.save('og.png')
        decoded.save('decoded.png')
        np.save('test.npy', decoded_imgs[idx])

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[idx])
    plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
