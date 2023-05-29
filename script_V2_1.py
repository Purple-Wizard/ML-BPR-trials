import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dropout, Input, Conv2DTranspose
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import cv2
import warnings
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)





dataset_path = "archive"
model_save_path = "models/best_model.h5"
num_images = 1000 # 176000
epochs = 100
img_size = (32, 32)
opt_trials = 1


def main():    
    original_images, processed_images = load_images(dataset_path, num_images)
    original_images = resize_images(original_images, img_size)
    processed_images = resize_images(processed_images, img_size)

    x_train, x_val, x_test = split_data(original_images)
    x_train_processed, x_val_processed, x_test_processed = split_data(processed_images)

    study = optuna.create_study(
        storage="sqlite:///optuna_vis.db", 
        direction="minimize",
        study_name="mask_ae_v2",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(
        trial, x_train, x_train_processed, x_val, x_val_processed, epochs), n_trials=opt_trials)

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.5f}")
    print(f"Best parameters: {best_trial.params}")

    best_model, best_batch_size, data_generator = create_model(best_trial)
    history = train_model(best_model, x_train, x_train_processed, x_val, x_val_processed, epochs, best_batch_size, data_generator)

    visualize_and_evaluate_results(best_model, x_test, history)
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")
    print(best_model.summary())


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

def create_simple_encoder(input_shape, num_layers, n_units, activations, dropout_rates):
    input_img = Input(shape=input_shape)
    x = input_img
    for i in range(num_layers):
        x = Conv2D(n_units[i], (3, 3), activation=activations[i], padding='same')(x)
        x = Dropout(dropout_rates[i])(x)
    encoded = x
    return Model(input_img, encoded)

def create_simple_decoder(input_shape, num_layers, n_units, activations, dropout_rates):
    encoded = Input(shape=input_shape)
    x = encoded
    for i in range(num_layers):
        x = Conv2DTranspose(n_units[i], (3, 3), activation=activations[i], padding='same')(x)
        x = Dropout(dropout_rates[i])(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(encoded, decoded)

def create_model(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    l2_weight = trial.suggest_loguniform("l2_weight", 1e-5, 1e-2)
    encoder_units = [
        trial.suggest_int("encoder_units_1", 512, 1024, 2048),
        trial.suggest_int("encoder_units_2", 256, 512, 1024),
        trial.suggest_int("encoder_units_3", 256, 512, 2048),
    ]
    print('encoder_units: ', encoder_units)
    decoder_units = [
        trial.suggest_int("decoder_units_1", 128, 256, 1024),
        trial.suggest_int("decoder_units_2", 256, 512, 2048),
        trial.suggest_int("decoder_units_3", 512, 1024, 2048),
    ]
    print('decoder_units: ', decoder_units)
    encoder_dropout_rates = [
        trial.suggest_float("encoder_dropout_rate_1", 0.1, 0.5),
        trial.suggest_float("encoder_dropout_rate_2", 0.1, 0.5),
        trial.suggest_float("encoder_dropout_rate_3", 0.1, 0.5),
    ]
    decoder_dropout_rates = [
        trial.suggest_float("decoder_dropout_rate_1", 0.1, 0.5),
        trial.suggest_float("decoder_dropout_rate_2", 0.1, 0.5),
        trial.suggest_float("decoder_dropout_rate_3", 0.1, 0.5),
    ]
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 3)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 3)
    decoder_activations = [trial.suggest_categorical(f"decoder_activation_{i}", ["relu", "elu", "tanh", "selu"]) for i in range(num_decoder_layers)]
    print('decoder_activations: ', decoder_activations)
    encoder_activations = [trial.suggest_categorical(f"encoder_activation_{i}", ["selu", "relu", "elu", "tanh"]) for i in range(num_encoder_layers)]
    print('encoder_activations: ', encoder_activations)
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
    input_shape = (h, w, 3)

    encoder = create_simple_encoder(input_shape, num_encoder_layers, encoder_units, encoder_activations, encoder_dropout_rates)
    decoder = create_simple_decoder(encoder.output_shape[1:], num_decoder_layers, decoder_units, decoder_activations, decoder_dropout_rates)

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

def train_model(model, x_train, x_train_processed, x_val, x_val_processed, epochs, batch_size, data_generator):
    print("Training the model...\n_________________________________________________________________")
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min',
        min_delta=0.001, cooldown=0, min_lr=0
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.001, 
        patience=6, 
        verbose=1, 
        mode='min', 
        restore_best_weights=True
    )

    history = model.fit(
        x=x_train, y=x_train_processed,
        epochs=epochs, 
        callbacks=[lr_scheduler, early_stopping],
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=[x_val, x_val_processed],
        validation_steps=len(x_val) // batch_size,
        verbose=1
    )

    return history

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
        ssims = [structural_similarity(x_test[i], decoded_imgs[i], data_range=1, multichannel=True, win_size=3) for i in range(len(x_test))]
        ssims_re = [structural_similarity(x_test_processed[i], decoded_imgs[i], data_range=1, multichannel=True, win_size=3) for i in range(len(x_test))]

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
        plt.title(f'Reconstructed\nAvg SSIM: {np.mean(ssims):.4f} \nSSIM_re :{np.mean(ssims_re):.4f}', fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def objective(trial, x_train, x_train_processed, x_val, x_val_processed, epochs):
    model, batch_size, data_generator = create_model(trial)
    model.summary()

    history = train_model(model, x_train, x_train_processed, x_val, x_val_processed, epochs, batch_size, data_generator)
    loss = history.history["val_loss"][-1]

    return loss


main()