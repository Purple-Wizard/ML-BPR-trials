import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Layer, Conv2D, Flatten, Dense, Reshape, Dropout, Conv2DTranspose
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)





dataset_path = "archive"
model_save_path = "models/best_model.h5"
num_images = 1000 # 176000
epochs = 100
img_size = (64, 64)
h, w = img_size
input_shape = (h, w, 3)


def main():    
    original_images, processed_images = load_images(dataset_path, num_images)
    original_images = resize_images(original_images, img_size)
    processed_images = resize_images(processed_images, img_size)

    x_train_processed, x_val_processed, x_test_processed = split_data(processed_images)
    
    encoder, z_mean, z_log_var, z = create_encoder(input_shape, 2)
    decoder = create_decoder(2)
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer ='rmsprop')
    vae.fit(x_train_processed, epochs = 100, batch_size = 100)
    
    visualize_and_evaluate_results(vae, x_test_processed, vae.history)

def load_images(path, num_images):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]
    

    original_images = [
        np.array(Image.open(file))
        for file in tqdm(image_files, desc="Loading original images", unit="images")
    ]

    processed_images = preprocess_images(original_images)

    return np.array(original_images), np.array(processed_images)

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
    filtered = apply_salt_and_pepper_filter(gray, 0.04, 3)
    log = cv2.GaussianBlur(filtered, (kernel_size, kernel_size), sigma)
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

def apply_salt_and_pepper_filter(image, prob, kernel_size):
    """
    Applies a salt and pepper filter to the input image.

    Args:
        image: Input image.
        prob: Probability of a pixel being set to salt or pepper.
        kernel_size: Size of the median filter kernel.

    Returns:
        Filtered image.
    """
    # Add salt and pepper noise
    noise = np.random.choice([0, 255], size=image.shape[:2], p=[prob, 1-prob])
    noise = noise.astype(np.uint8)
    img_noise = cv2.add(image, noise, dtype=cv2.CV_8UC3)

    # Apply median filter
    img_filtered = cv2.medianBlur(img_noise, kernel_size)

    return img_filtered
    

def preprocess_images(images):
    processed_images = []

    for img in tqdm(images, desc="Preprocessing images", unit="images"):
        log = apply_log_filter(img, kernel_size=5, sigma=0.5)
        sobel = apply_sobel_filter(img, kernel_size=3)
        img = img.astype(np.float32)
        log = log.astype(np.float32)
        log_color = cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.6, log_color, 0.2, 0)
        img = cv2.addWeighted(img, 0.8, cv2.cvtColor(sobel.astype(np.float32), cv2.COLOR_GRAY2BGR), 0.2, 0)
        processed_images.append(img)

    return np.array(processed_images, dtype=images[0].dtype)

def create_image_data_generator(
                    rotation_range=20, 
                    width_shift_range=0.2, 
                    height_shift_range=0.2, 
                    shear_range=0.2, 
                    zoom_range=0.2, 
                    horizontal_flip=True
                ):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode="constant",
        cval=0.0
    )

def split_data(images):
    print("Splitting data to train, val, and test...\n_________________________________________________________________")
    images = np.array(images)
    x_train, x_temp = train_test_split(images, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)
    return x_train, x_val, x_test


class Sampling(Layer):
 
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape =(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



def create_encoder(input_shape, latent_dim):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, 3, activation ="elu", strides = 2, padding ="same")(input_img)
    x = Conv2D(64, 3, activation ="relu", strides = 2, padding ="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation ="relu")(x)
    z_mean = Dense(latent_dim, name ="z_mean")(x)
    z_log_var = Dense(latent_dim, name ="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(input_img, [z_mean, z_log_var, z], name ="encoder")
    print(encoder.summary())
    return encoder


def create_decoder(latent_dim):
    encoded = keras.Input(shape =(latent_dim, ))
    x = Dense(8 * 8 * 64, activation ="relu")(encoded)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(32, 3, activation ="selu", strides = 2, padding ="same")(x)
    x = Conv2DTranspose(64, 3, activation ="relu", strides = 2, padding ="same")(x)
    x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(3, 3, activation ="sigmoid", padding ="same")(x)
    decoder = Model(encoded, decoder_outputs, name ="decoder")
    print(decoder.summary())
    return decoder
    

# this class takes encoder and decoder models and
# define the complete variational autoencoder architecture
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
 
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            encoder, z_mean, z_log_var, z = create_encoder(input_shape, 2)
            decoder = create_decoder(2)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }



def visualize_and_evaluate_results(model, x_test, history):
    x_test_processed = preprocess_images(x_test)
    decoded_imgs = model.predict(x_test_processed)

    n = 10
    plt.figure(figsize=(21, 7))
    
    # Plotting the training and validation loss curves
    ax = plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    for i in tqdm(range(n), desc="Visualising and evaluating results", unit="images"):
       # psnrs = [peak_signal_noise_ratio(x_test[i], decoded_imgs[i], data_range=1) for i in range(len(x_test))]
        RL = [reconstruction_loss(x_test[i], decoded_imgs[i]) for i in range(len(x_test))]
        SS = [structural_similarity(x_test_processed[i], decoded_imgs[i], data_range=1, multichannel=True, win_size=3) for i in range(len(x_test))]

        # Plotting the original images
        ax = plt.subplot(4, n, i + n + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Plotting the processed images
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(x_test_processed[i])
        plt.title("Processed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plotting the reconstructed images
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(decoded_imgs[i])
        plt.title(f'Reconstructed\nAvg RECON_LOSS: {np.mean(RL):.4f} \nSTRUC_SIMI :{np.mean(SS):.4f}', fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def reconstruction_loss(original, reconstructed):
    mse_loss = tf.keras.losses.mean_squared_error(original, reconstructed)
    return mse_loss

if __name__ == '__main__':
  main()