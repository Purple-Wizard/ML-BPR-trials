import tensorflow as tf
from preprocess import load_images, postprocess_images
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Layer, UpSampling2D, MaxPooling2D, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

class HashingLayer(Layer):
    def __init__(self, num_bits, **kwargs):
        self.num_bits = num_bits
        super(HashingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.num_bits),
                                      initializer='uniform',
                                      trainable=True)
        super(HashingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # Hashing operation: dot product with kernel followed by sign function
        hash_code = tf.sign(tf.matmul(x, self.kernel))
        return hash_code

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_bits)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_bits": self.num_bits,
        })
        return config

class Autoencoder:
    def __init__(self, latent_dim, num_hash_bits):
        self.latent_dim = latent_dim
        self.num_hash_bits = num_hash_bits
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self._autoencoder = tf.keras.Sequential([self.encoder, self.decoder])

    @property
    def autoencoder(self):
        return self._autoencoder
    
    def build_encoder(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim),
            HashingLayer(self.num_hash_bits)
        ])
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_hash_bits,)),
            tf.keras.layers.Dense(8 * 8 * 256),
            tf.keras.layers.Reshape((8, 8, 256)),
            tf.keras.layers.Conv2DTranspose(256, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(128, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same', strides=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(3, (5, 5), activation='sigmoid', padding='same')
        ])
        return decoder

autoencoder = Autoencoder(latent_dim=256, num_hash_bits=2048)
autoenc = autoencoder.autoencoder
x_train, x_val, original_test, x_train_processed, processed_val, x_test_processed, min_max_vals, noise_cords = load_images('archive', 1000)

lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min',
        min_delta=0.001, cooldown=3, min_lr=0
    )
early_stopping = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.0001, 
        patience=20, 
        verbose=1, 
        mode='min', 
        restore_best_weights=True
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
autoenc.compile(optimizer, loss='BinaryCrossentropy')

history = autoenc.fit(x_train_processed, x_train_processed, epochs=300, batch_size=32, validation_data=(processed_val, processed_val),
        steps_per_epoch=len(x_train_processed) // 32, verbose=1, callbacks=[lr_scheduler, early_stopping])

autoencoder.encoder.save('encoder.h5', save_format='tf')
autoencoder.decoder.save('decoder.h5', save_format='tf')

n = 10
random_indices = random.sample(range(x_test_processed.shape[0]), n)
plt.figure(figsize=(21, 7))
ax = plt.subplot(3, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
for i, idx in enumerate(random_indices):
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(x_test_processed[idx].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    orig_size = x_test_processed[idx].size * x_test_processed[idx].itemsize

    # display encoded
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    z = autoencoder.encoder.predict(x_test_processed[idx].reshape(1, 64, 64, 3))
    decoded_img = autoencoder.decoder.predict(z)
    # post_img = postprocess_images([decoded_img.reshape(64, 64, 3)], min_max_vals, noise_cords)
    ssims = [structural_similarity(x_test_processed[idx], decoded_img.reshape(64, 64, 3), data_range=1, multichannel=True, win_size=3) for i in range(len(x_test_processed))]
    plt.imshow(decoded_img.reshape(64, 64, 3))
    plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
    plt.jet()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    enc_size = z.size * z.itemsize
    dec_size = decoded_img.size * decoded_img.itemsize
    print(f'Encoded: {enc_size} bytes | Decoded: {dec_size} bytes')

plt.show()
