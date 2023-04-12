import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import optuna
from keras.regularizers import l2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)
# tf.debugging.set_log_device_placement(True)

dataset_path = "dataset_128x128"
num_images = 20000 # 176000
epochs = 100
img_size = (64, 64)
opt_trials = 50

def main():    
    images = load_images(dataset_path, num_images)
    images = preprocess_images(images, img_size)
    x_train, x_val, x_test = split_data(images)

    # Optimize using Optuna
    study = optuna.create_study(
        storage="sqlite:///optuna_vis.db", 
        direction="minimize",
        study_name="v1.9",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(
        trial, x_train, x_val, epochs), n_trials=opt_trials)

    # Display best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.5f}")
    print(f"Best parameters: {best_trial.params}")

    # Train the model with the best hyperparameters
    best_model, best_batch_size, data_generator = create_model(best_trial)
    history = train_model(best_model, x_train, x_val, epochs, best_batch_size, data_generator)

    # Evaluate and visualize the model
    evaluate_model(history)
    visualize_results(best_model, x_test)
    model_save_path = "best_model.h5"
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")

def load_images(path, num_images):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]

    images = [
        np.array(Image.open(file))
        for file in tqdm(image_files, desc="Loading Images", unit="images")
    ]

    return np.array(images)

def preprocess_images(images, image_size=None):
    processed_images = []
    for image in tqdm(images, desc="Preprocessing images", unit="images"):
        img = Image.fromarray(image)
        
        if image_size is not None:
            img = ImageOps.fit(img, image_size, Image.LANCZOS)
        
        processed_images.append(np.array(img))
    
    processed_images = np.array(processed_images)
    processed_images = processed_images.astype("float32") / 255.0
    return processed_images

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

def apply_pca(x_train, x_val, n_components):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train.reshape(x_train.shape[0], -1))
    x_val_pca = pca.transform(x_val.reshape(x_val.shape[0], -1))

    return x_train_pca, x_val_pca

def split_data(images):
    print("Splitting data...")
    x_train, x_temp = train_test_split(images, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)
    return x_train, x_val, x_test

def create_pretrained_encoder(input_shape):
    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    return encoder

def create_decoder(input_shape, n_filters, l2_weight, activations, dropout_rates, pooling_strategy):
    encoded = Input(shape=input_shape)
    x = encoded

    x = Conv2D(n_filters[0], (3, 3), activation=activations[0], padding="same", kernel_regularizer=l2(l2_weight))(x)
    x = Dropout(dropout_rates[0])(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(n_filters[1], (3, 3), activation=activations[1], padding="same", kernel_regularizer=l2(l2_weight))(x)
    x = Dropout(dropout_rates[1])(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(n_filters[2], (3, 3), activation=activations[2], padding="same", kernel_regularizer=l2(l2_weight))(x)
    x = Dropout(dropout_rates[2])(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(n_filters[3], (3, 3), activation=activations[3], padding="same", kernel_regularizer=l2(l2_weight))(x)
    x = Dropout(dropout_rates[3])(x)
    x = UpSampling2D((2, 2))(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(n_filters[4], (3, 3), activation=activations[4], padding="same", kernel_regularizer=l2(l2_weight))(x)
    x = Dropout(dropout_rates[4])(x)

    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    decoder = Model(encoded, decoded)
    return decoder

def create_model(trial):
    print("Creating autoencoder model...")

    # Retrieve hyperparameters from the trial
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    l2_weight = trial.suggest_loguniform("l2_weight", 1e-5, 1e-2)
    activations = [
        trial.suggest_categorical("activation_1", ["relu", "elu"]),
        trial.suggest_categorical("activation_2", ["relu", "elu"]),
        trial.suggest_categorical("activation_3", ["relu", "elu"]),
        trial.suggest_categorical("activation_4", ["relu", "elu"]),
        trial.suggest_categorical("activation_5", ["relu", "elu"]),
    ]
    dropout_rates = [
        trial.suggest_float("dropout_rate_1", 0.1, 0.5),
        trial.suggest_float("dropout_rate_2", 0.1, 0.5),
        trial.suggest_float("dropout_rate_3", 0.1, 0.5),
        trial.suggest_float("dropout_rate_4", 0.1, 0.5),
        trial.suggest_float("dropout_rate_5", 0.1, 0.5),
    ]

    n_filters = [
        trial.suggest_int("n_filters_1", 16, 128),
        trial.suggest_int("n_filters_2", 16, 128),
        trial.suggest_int("n_filters_3", 16, 128),
        trial.suggest_int("n_filters_4", 16, 128), 
        trial.suggest_int("n_filters_5", 16, 128)
    ]
    rotation_range = trial.suggest_int("rotation_range", 0, 45)
    width_shift_range = trial.suggest_float("width_shift_range", 0.0, 0.5)
    height_shift_range = trial.suggest_float("height_shift_range", 0.0, 0.5)
    shear_range = trial.suggest_float("shear_range", 0.0, 0.5)
    zoom_range = trial.suggest_float("zoom_range", 0.0, 0.5)
    
    pooling_strategy = [
        trial.suggest_categorical(f"pooling_strategy_layer_{i}", ["MaxPooling2D", "AveragePooling2D"]) for i in range(3)
    ]

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    
    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    h, w = img_size

    encoder = create_pretrained_encoder((h, w, 3))
    encoder_output_shape = encoder.output_shape[1:]
    decoder = create_decoder(encoder_output_shape, n_filters, l2_weight, activations, dropout_rates, pooling_strategy)

    data_generator = create_image_data_generator(
        rotation_range=rotation_range, 
        width_shift_range=width_shift_range, 
        height_shift_range=height_shift_range, 
        shear_range=shear_range, 
        zoom_range=zoom_range
    )

    model = Sequential([encoder, decoder])
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model, batch_size, data_generator

def train_model(model, x_train, x_val, epochs, batch_size, data_generator):
    print("Training the model...")
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=2, mode='min',
        min_delta=0.001, cooldown=0, min_lr=0
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.001, 
        patience=6, 
        verbose=2, 
        mode='min', 
        restore_best_weights=True
    )
    train_generator = data_generator.flow(x_train, x_train, batch_size=batch_size)

    history = model.fit(
        train_generator, 
        epochs=epochs, 
        validation_data=(x_val, x_val), 
        callbacks=[lr_scheduler, early_stopping]
    )

    return history

# Evaluate the model
def evaluate_model(history):
    print("Evaluating the model...")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Visualize the results
def visualize_results(model, x_test):
    print("Visualizing results...")
    decoded_imgs = model.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def objective(trial, x_train, x_val, epochs):
    model, batch_size, data_generator = create_model(trial)
    model.summary()

    # Optimize PCA components
    # n_components = trial.suggest_int("n_components", 2, 64)
    # x_train_pca, x_val_pca = apply_pca(x_train, x_val, n_components=n_components)

    # Train the model
    history = train_model(model, x_train, x_val, epochs, batch_size, data_generator)

    loss = history.history["val_loss"][-1]

    return loss


if __name__ == "__main__":
    main()