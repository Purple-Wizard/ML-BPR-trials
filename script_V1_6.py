import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import optuna
from keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')
# tf.debugging.set_log_device_placement(True)

def main():
    dataset_path = "dataset_128x128"
    num_images = 40000 # 176000
    epochs = 100
    img_size = (64, 64)
    
    # Load and preprocess images
    images = load_images(dataset_path, num_images)
    images = preprocess_images(images, img_size)
    x_train, x_val, x_test = split_data(images)

    # Optimize using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(
        trial, x_train, x_val, epochs), n_trials=60)

    # Display best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.5f}")
    print(f"Best parameters: {best_trial.params}")

    # Train the model with the best hyperparameters
    best_model, best_batch_size = create_model(best_trial)
    history = train_model(best_model, x_train, x_val, epochs, best_batch_size)

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
                images.append(np.array(image))
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


def split_data(images):
    print("Splitting data...")
    x_train, x_temp = train_test_split(images, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)
    return x_train, x_val, x_test

def create_encoder(n_filters, l2_weight, activations):
    encoder = Sequential()
    encoder.add(Conv2D(n_filters[0], (3, 3), activation=activations[0], padding="same", input_shape=(
        64, 64, 3), kernel_regularizer=l2(l2_weight)))
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
        8, 8, n_filters[2]), kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(n_filters[1], (3, 3), activation=activations[1],
                padding="same", kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(n_filters[0], (3, 3), activation=activations[2],
                padding="same", kernel_regularizer=l2(l2_weight)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same"))
    return decoder

def create_model(trial):
    print("Creating autoencoder model...")

    # Retrieve hyperparameters from the trial
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    l2_weight = trial.suggest_loguniform("l2_weight", 1e-5, 1e-2)
    activations = [
        trial.suggest_categorical(f"activation_layer_{i}", ["relu", "elu", "sigmoid", "tanh", "selu", "softmax"]) for i in range(3)
    ]

    n_filters = [
        trial.suggest_int(f"n_filters_layer_{i}", 16, 64) for i in range(3)
    ]

    encoder = create_encoder(n_filters, l2_weight, activations)
    decoder = create_decoder(n_filters, l2_weight, activations)

    model = Sequential([encoder, decoder])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy")
    return model, batch_size

def train_model(model, x_train, x_val, epochs, batch_size):
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
    history = model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(
        x_val, x_val), callbacks=[lr_scheduler, early_stopping])
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
    model, batch_size = create_model(trial)
    model.summary()

    history = train_model(model, x_train, x_val, epochs, batch_size)
    loss = history.history["val_loss"][-1]

    return loss


if __name__ == "__main__":
    main()