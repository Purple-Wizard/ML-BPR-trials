import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from preprocess import load_images

x_train, x_val, original_test, x_train_processed, processed_val, x_test_processed = load_images('archive', 1000)
x_train_processed = np.expand_dims(x_train_processed, axis=-1)

# Set the random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the dimensions of the latent space
latent_dim = 16

encoder_inputs = tf.keras.Input(shape=(64, 64, 3))  # Change shape to (64, 64, 1) for grayscale images
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

# Calculate the mean and variance for the latent space
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim),
                                            mean=0.0, stddev=1.0)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

# Sample from the latent space
z = layers.Lambda(sampling)([z_mean, z_log_var])

# Define the encoder model
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(256, activation="relu")(latent_inputs)
x = layers.Dense(8 * 8 * 64, activation="relu")(x)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)  # Upsampling layer
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)   # Upsampling layer
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)   # Upsampling layer
x = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)  # Output with 3 channels

# Define the decoder model
decoder = tf.keras.Model(latent_inputs, x, name="decoder")
decoder.summary()

# VAE model
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, vae_outputs, name="vae")

# Define the VAE loss function
def vae_loss(inputs, outputs):
    reconstruction_loss = tf.keras.backend.mean(
        tf.keras.losses.binary_crossentropy(inputs, outputs), axis=(1, 2)
    )
    kl_loss = -0.5 * tf.keras.backend.mean(
        1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1
    )
    return reconstruction_loss + kl_loss


# Load and preprocess the images (you mentioned you have this code)

# Compile the VAE model
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)

# Train the VAE model
vae.fit(x_train_processed, x_train_processed, batch_size=128, epochs=10)
