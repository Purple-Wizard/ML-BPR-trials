from hashing_model import HashingLayer
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load the model
encoder = tf.keras.models.load_model('models/encoder.h5', custom_objects={'HashingLayer': HashingLayer})
decoder = tf.keras.models.load_model('models/decoder.h5')

# Print the model architecture
encoder.summary()
decoder.summary()

# Plot the model
plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)
