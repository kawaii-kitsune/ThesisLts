import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
import torch.optim as optim

# Load the segmentation model (CNN)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load pre-trained segmentation model (Replace with actual model)
segmentation_model = SimpleUNet()
segmentation_model.eval()

# Load Transformer model for classification
model_name = "google/vit-base-patch16-224"
classifier = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

st.title("Segmentation & Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for segmentation
    transform_seg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform_seg(image).unsqueeze(0)  # Add batch dim

    # Run segmentation
    with torch.no_grad():
        segmented_mask = segmentation_model(img_tensor)
    
    # Convert mask to image format
    segmented_mask_np = segmented_mask.squeeze().numpy()
    segmented_mask_np = (segmented_mask_np * 255).astype(np.uint8)
    
    st.image(segmented_mask_np, caption="Segmented Mask", use_column_width=True)

    # Preprocess for classification
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Run classification
    with torch.no_grad():
        outputs = classifier(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()

    st.write(f"Predicted Class: {classifier.config.id2label[predicted_class]}")
