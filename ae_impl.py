import tensorflow as tf
import tensorflow_probability as tfp
from preprocess import load_images
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Layer, UpSampling2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import random

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
    
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            reconstruction = self(data)
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]
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

def encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (5, 5), activation="elu", padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (5, 5), activation="elu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(256, (5, 5), activation="elu", padding="same")(x)
    x = MaxPooling2D((5, 5), padding="same")(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z])

def decoder(latent_dim):
    decoder = Sequential()
    decoder.add(Dense(8*8*256, input_shape=(latent_dim,)))
    decoder.add(Reshape((8, 8, 256)))
    decoder.add(Conv2D(256, (5, 5), activation="elu", padding="same"))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(128, (5, 5), activation="elu", padding="same"))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(64, (5, 5), activation="elu", padding="same"))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(3, (5, 5), activation="sigmoid", padding="same"))
    return decoder

dataset_path = "archive"
num_images = 1000
epochs = 150

x_train, x_val, original_test, x_train_processed, processed_val, x_test_processed = load_images(dataset_path, num_images)

input_shape = x_train.shape[1:]
latent_dim = 256

enc = encoder((64, 64, 3), latent_dim)
dec = decoder(latent_dim)
autoenc = VAE(enc, dec)

lr_scheduler = ReduceLROnPlateau(
        monitor='reconstruction_loss', factor=0.1, patience=10, verbose=1, mode='min',
        min_delta=0.001, cooldown=3, min_lr=0
    )
early_stopping = EarlyStopping(
        monitor="reconstruction_loss", 
        min_delta=0.0001, 
        patience=20, 
        verbose=1, 
        mode='min', 
        restore_best_weights=True
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
autoenc.compile(optimizer)

history = autoenc.fit(x_train_processed, x_train_processed, epochs=epochs, batch_size=64, validation_data=[x_test_processed, x_test_processed],
        validation_steps=len(x_test_processed) // 64, steps_per_epoch=len(x_train_processed) // 64, verbose=1, callbacks=[lr_scheduler, early_stopping])

enc.save('encoder.h5', save_format='tf')
dec.save('decoder.h5', save_format='tf')

n = 10
random_indices = random.sample(range(x_val.shape[0]), n)
plt.figure(figsize=(21, 7))
ax = plt.subplot(3, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['reconstruction_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
for i, idx in enumerate(random_indices):
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(x_val[idx].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    orig_size = x_val[idx].size * x_val[idx].itemsize

    # display encoded
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    z_mean, z_log_var, z = enc.predict(x_val[idx].reshape(1, 64, 64, 3))
    # Sample a point from the latent space
    epsilon = np.random.normal(0, 1, size=z_mean.shape)
    z = z_mean + np.exp(0.5 * z_log_var) * epsilon
    decoded_img = dec.predict(z)
    ssims = [structural_similarity(x_val[idx], decoded_img.reshape(64, 64, 3), data_range=1, multichannel=True, win_size=3) for i in range(len(x_val))]
    plt.imshow(decoded_img.reshape(64, 64, 3))
    plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    enc_size = z.size * z.itemsize
    dec_size = decoded_img.size * decoded_img.itemsize
    print(f'Encoded: {enc_size} bytes | Decoded: {dec_size} bytes')

plt.show()
