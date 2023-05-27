import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from skimage.metrics import structural_similarity
import random
import numpy as np
import matplotlib.pyplot as plt

from hashing_model import HashingLayer
from preprocessv2 import load_images

data = load_images('dataset_128x128', 1000)

train_data_arr = data['train']
test_data_arr = data['test']
validation_data_arr = data['val']

train_data_arr = train_data_arr.batch(32)
test_data_arr = test_data_arr.batch(32)
validation_data_arr = validation_data_arr.batch(32)

# def create_encoder(input_shape=(128, 128, 3)):
#     input_img = Input(shape=input_shape)
#     a1 = tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='same')(input_img)
#     a2 = tf.keras.layers.Activation('relu')(a1)
#     a3 = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding='same')(a2)
#     a4 = tf.keras.layers.Activation('relu')(a3)
#     a5 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same') (a4)
#     a6 = tf.keras.layers.Activation('relu')(a5)
#     a7 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(a6)
#     a8 = tf.keras.layers.Activation('relu')(a7)
#     skip_0 = tf.keras.layers.Add()([a8, a6])
#     a9 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(skip_0)
#     a10 = tf.keras.layers.Activation('relu')(a9)
#     a11 = tf.keras.layers.Conv2D(3, (3,3),strides=1, padding='same')(a10)
#     a12 = tf.keras.layers.Activation('relu')(a11)
    
#     return Model(input_img, a12)

# def create_decoder(encoder):
#     input_shape = encoder.output.shape[1:]
#     decoder_input = Input(shape=input_shape)
#     a13 = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=1, padding='same')(decoder_input)
#     a14 = tf.keras.layers.Activation('relu')(a13)
#     a15 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=2, padding='same')(a14)
#     a16 = tf.keras.layers.Activation('relu')(a15)
#     a17 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=1, padding='same')(a16)
#     a18 = tf.keras.layers.Activation('relu')(a17)
#     a19 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=1, padding='same')(a18)
#     a20 = tf.keras.layers.Activation('relu')(a19)
#     skip_1 = tf.keras.layers.Add()([a18, a20])
#     a21 = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=1, padding='same')(skip_1)
#     a22 = tf.keras.layers.Activation('relu')(a21)

#     return Model(decoder_input, a22)

def ConvBlock(x, filters, strides, padding='same'):
    x = Conv2D(filters, (3,3), strides=strides, padding=padding)(x)
    x = Activation('relu')(x)
    return x

def ConvTransposeBlock(x, filters, strides, padding='same'):
    x = Conv2DTranspose(filters, (3,3), strides=strides, padding=padding)(x)
    x = Activation('relu')(x)
    return x

def create_encoder(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)
    a1 = ConvBlock(input_img, 32, 1)
    a3 = ConvBlock(a1, 64, 2)
    a5 = ConvBlock(a3, 128, 1)
    a7 = ConvBlock(a5, 128, 1)
    skip_0 = Add()([a7, a5])
    a9 = ConvBlock(skip_0, 64, 1)
    a11 = ConvBlock(a9, 3, 1)
    a12 = HashingLayer(2)(Model(input_img, a11).output)
    
    return tf.keras.Model(input_img, a12)

def create_decoder(hashing_model):
    input_shape = hashing_model.output.shape[1:]
    decoder_input = Input(shape=input_shape)
    a13 = ConvTransposeBlock(decoder_input, 32, 1)
    a15 = ConvTransposeBlock(a13, 128, 2)
    a17 = ConvTransposeBlock(a15, 64, 1)
    a19 = ConvTransposeBlock(a17, 64, 1)
    skip_1 = Add()([a17, a19])
    a21 = ConvTransposeBlock(skip_1, 3, 1)

    return tf.keras.Model(decoder_input, a21)

encoder = create_encoder()
decoder = create_decoder(encoder)

model = Model(encoder.input, decoder(encoder.output))

lr_scheduler = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5,
                    verbose=1,
                    mode='min',
                    min_delta=0.001,
                    cooldown=3,
                    min_lr=1e-6
                )
early_stopping = EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.0001,
                    patience=12,
                    verbose=1,
                    mode='min',
                    restore_best_weights=True
                )

encoder.summary()
decoder.summary()

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mae')
history_comp = model.fit(
    train_data_arr,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    validation_data=(validation_data_arr),
    verbose=1,
    callbacks=[lr_scheduler, early_stopping]
)

encoder.save('models/encoder.h5')
decoder.save('models/decoder.h5')

encoded = encoder.predict(test_data_arr)
decoded = decoder.predict(encoded)

n = 10
test_iter = next(iter(test_data_arr))
random_indices = random.sample(range(test_iter[0].shape[0]), n)
plt.figure(figsize=(21, 7))
ax = plt.subplot(3, 1, 1)
plt.plot(history_comp.history['loss'], label='Training Loss')
plt.plot(history_comp.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
for i, idx in enumerate(random_indices):
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(test_iter[0][idx])
    ssims = [structural_similarity(test_iter[1][idx].numpy().reshape(128, 128, 3), decoded[idx].reshape(128, 128, 3), data_range=1, multichannel=True, win_size=3) for i in range(len(decoded))]
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded[idx])
    plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    og_size = test_iter[0][idx].numpy().size * test_iter[0][idx].numpy().itemsize
    enc_size = encoded[idx].size * encoded[idx].itemsize
    dec_size = decoded[idx].size * decoded[idx].itemsize
    print(f'Original: {og_size} bytes | Encoded: {enc_size} bytes | Decoded: {dec_size} bytes')

plt.show()
