import sys
import os
import torch

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models.unet.model as Unet
import dataloaders.cnnDataLoader as cnnDataLoader
import torch.optim as optim
import torch.nn as nn
from keras.callbacks import EarlyStopping

data_dir = 'C:\\Users\\babis\\Documents\\GitHub\\ThesisLts\\data\\cnn_dataset'
unet_train, unet_valid = cnnDataLoader.get_dataloaders(data_dir)

model= Unet.UNet()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(unet_train, validation_data=unet_valid, epochs=20, callbacks=[EarlyStopping(patience=3)])

# model.save("unet_model.h5")  # Saves in HDF5 format
