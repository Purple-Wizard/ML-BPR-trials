import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.models import Model
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import warnings
from preprocess import load_images
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)


dataset_path = "archive"
num_images = 1000 # 176000
epochs = 100
img_size = (64, 64)
h, w = img_size
input_shape = (h, w, 3)
hash_bits = 16
latent_dim = 2



# load image
X_train, x_val, X_test, X_train_preprocess, X_val_preprocess, X_test_preprocess = load_images(dataset_path, num_images)

print(X_train_preprocess.shape)

# this sampling layer is the bottleneck layer of variational autoencoder,
# it uses the output from two dense layers z_mean and z_log_var as input,
# convert them into normal distribution and pass them to the decoder layer
class Sampling(Layer):
 
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape =(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
    
# Define Encoder Model
input_img = Input(shape=input_shape)
x = Conv2D(32, 3, activation ="elu", strides = 2, padding ="same")(input_img)
x = Conv2D(64, 3, activation ="relu", strides = 2, padding ="same")(x)
x = Flatten()(x)
x = Dense(hash_bits, activation ="sigmoid")(x)
z_mean = Dense(latent_dim, name ="z_mean")(x)
z_log_var = Dense(latent_dim, name ="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(input_img, [z_mean, z_log_var, z], name ="encoder")
print(encoder.summary())




# Define Decoder Model
encoded = keras.Input(shape=(latent_dim,))
x = Dense(8 * 8 * 64, activation ="relu")(encoded)
x = Reshape((8, 8, 64))(x)
x = Conv2DTranspose(32, 3, activation ="selu", strides = 2, padding ="same")(x)
x = Conv2DTranspose(64, 3, activation ="relu", strides = 2, padding ="same")(x)
x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = Conv2DTranspose(3, 3, activation ="sigmoid", padding ="same")(x)
decoder = Model(encoded, decoder_outputs, name ="decoder")
print(decoder.summary())





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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
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
        



# compile and train the model
vae = VAE(encoder, decoder)
vae.compile(optimizer ='rmsprop')
vae.fit(X_train_preprocess, epochs = 100, batch_size = 100)



def visualize_and_evaluate_results(model, x_test, history):

    n = 10
    plt.figure(figsize=(21, 7))

    # Plotting the training and validation loss curves
    ax = plt.subplot(3, 1, 1)
    plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(history.history['kl_loss'], label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    z_mean, z_log_var, z = model.encoder.predict(X_test_preprocess)
    decoded_img = model.decoder.predict(z)

    for i in range(1, 11):
        RL = [reconstruction_loss(x_test[i], decoded_img[i]) for i in range(len(x_test))]
        SS = [structural_similarity(X_test_preprocess[i], decoded_img[i], data_range=1, multichannel=True, win_size=3) for i in range(len(x_test))]

        # Plotting the original images
        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plotting the reconstructed images
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_img[i])
        plt.title(f'Reconstructed\nAvg RECON_LOSS: {np.mean(RL):.4f} \nSTRUC_SIMI :{np.mean(SS):.4f}', fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        i += 1

    plt.show()


def reconstruction_loss(original, reconstructed):
    mse_loss = tf.keras.losses.mean_squared_error(original, reconstructed)
    return mse_loss

visualize_and_evaluate_results(vae, X_test_preprocess, vae.history)