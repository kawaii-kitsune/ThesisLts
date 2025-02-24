import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import cv2
import os

# Define the U-Net model
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = Concatenate()([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = Concatenate()([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(2, (1, 1), activation='softmax')(c5)

    model = Model(inputs, outputs)
    return model

# Compile the model
model = unet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the data (example placeholders)
# Updated preprocessing to convert mask values to 0 and 1
def preprocess_data(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []

    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, os.path.splitext(file_name)[0] + ".png")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: Could not read {file_name}. Skipping.")
            continue

        image = cv2.resize(image, image_size)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

        # Convert mask values: 155 becomes 1, 0 remains as 0
        mask = (mask == 155).astype(np.uint8)  # 155 -> 1, background -> 0

        images.append(image)
        masks.append(mask)

    if not images or not masks:
        raise ValueError("No valid images or masks found. Please check your dataset paths.")

    images = np.array(images) / 255.0  # Normalize images to [0, 1]
    masks = np.array(masks)           # Masks should remain as integers

    return images, masks

# Updated segmentation saving
def segment_and_save(image_path, model, output_dir):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    resized_image = cv2.resize(image, (256, 256)) / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)

    prediction = model.predict(resized_image)[0]
    prediction = np.argmax(prediction, axis=-1)
    prediction = cv2.resize(prediction.astype('uint8'), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # Save segmented images
    leaf_mask = (prediction == 1).astype('uint8') * 255
    cv2.imwrite(os.path.join(output_dir, "leaf.png"), leaf_mask)




# Example dataset directories
image_dir = "/home/babinos/Desktop/ThesisCode/data/Plant segmentation ready/train/images"
mask_dir = "/home/babinos/Desktop/ThesisCode/data/Plant segmentation ready/train/masks"

# Prepare data
X, y = preprocess_data(image_dir, mask_dir)
# Ensure data shapes are correct
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}, unique values: {np.unique(y)}")

# checkpoint = ModelCheckpoint("leaf_branch_segmentation_model_best.h5", save_best_only=True, monitor="val_loss")
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, callbacks=[early_stopping])


model.save("/home/babinos/Desktop/ThesisCode/models/leaf_branch_segmentation_model.h5")

# Example usage
segment_and_save("/home/babinos/Desktop/ThesisCode/data/Plant segmentation ready/validation/images/T02_Box007_2017-09-04T07-12-53-482.png", model, "/home/babinos/Desktop/ThesisCode/data/outputImages")
