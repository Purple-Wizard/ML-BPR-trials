import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import optuna
from keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)
# tf.debugging.set_log_device_placement(True)

def main():
    dataset_path = "dataset_128x128"
    num_images = 50000 # 176000
    epochs = 100
    img_size = (90, 90)
    opt_trials = 50
    
    # Load and preprocess images
    images = load_images(dataset_path, num_images)
    images = preprocess_images(images, img_size)
    x_train, x_val, x_test = split_data(images)

    # Optimize using Optuna
    study = optuna.create_study(
        storage="sqlite:///optuna_vis.db", 
        direction="minimize",
        study_name="v1.7",
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

def load_images(path, num_images):
    print("Loading images...")
    images = []
    image_count = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.png'):
                if image_count >= num_images:
                    break
                image = Image.open(os.path.join(root, file))
                try:
                    images.append(np.array(image))
                except:
                    print(file)
                image_count += 1
    return np.array(images)

# Preprocessing
def preprocess_images(images, image_size=None):
    print("Preprocessing images...")
    
    processed_images = []
    for image in images:
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

def create_encoder(n_filters, l2_weight, activations):
    encoder = Sequential()
    encoder.add(Conv2D(n_filters[0], (3, 3), activation=activations[0], padding="same", input_shape=(
        90, 90, 3), kernel_regularizer=l2(l2_weight)))
    encoder.add(MaxPooling2D((2, 2), padding="same"))
    encoder.add(Conv2D(n_filters[1], (3, 3), activation=activations[1],
                padding="same", kernel_regularizer=l2(l2_weight)))
    encoder.add(MaxPooling2D((2, 2), padding="same"))
    encoder.add(Conv2D(n_filters[2], (3, 3), activation=activations[2],
                padding="same", kernel_regularizer=l2(l2_weight)))
    encoder.add(MaxPooling2D((2, 2), padding="same"))
    return encoder

def create_decoder(n_filters, l2_weight, activations):
    decoder = Sequential()
    decoder.add(Conv2D(n_filters[2], (3, 3), activation=activations[0], padding="same", input_shape=(
        12, 12, n_filters[2]), kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(n_filters[1], (3, 3), activation=activations[1],
                padding="same", kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(n_filters[0], (3, 3), activation=activations[2],
                padding="same", kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(3, (7, 7), activation="sigmoid", padding="valid"))
    return decoder

def create_model(trial):
    print("Creating autoencoder model...")

    # Retrieve hyperparameters from the trial
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    l2_weight = trial.suggest_loguniform("l2_weight", 1e-5, 1e-2)
    activations = [
        trial.suggest_categorical(f"activation_layer_{i}", ["relu", "elu", "sigmoid", "tanh", "selu"]) for i in range(3)
    ]

    n_filters = [
        trial.suggest_int(f"n_filters_layer_{i}", 16, 90) for i in range(3)
    ]
    rotation_range = trial.suggest_int("rotation_range", 0, 45)
    width_shift_range = trial.suggest_float("width_shift_range", 0.0, 0.5)
    height_shift_range = trial.suggest_float("height_shift_range", 0.0, 0.5)
    shear_range = trial.suggest_float("shear_range", 0.0, 0.5)
    zoom_range = trial.suggest_float("zoom_range", 0.0, 0.5)

    data_generator = create_image_data_generator(
        rotation_range=rotation_range, 
        width_shift_range=width_shift_range, 
        height_shift_range=height_shift_range, 
        shear_range=shear_range, 
        zoom_range=zoom_range
    )

    encoder = create_encoder(n_filters, l2_weight, activations)
    decoder = create_decoder(n_filters, l2_weight, activations)

    model = Sequential([encoder, decoder])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy")
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