# Install necessary libraries
!pip install torch torchvision transformers
!pip install pytesseract
!pip install pdf2image
!pip install opencv-python
!pip install scikit-learn
!pip install easyocr
!pip install pyspellchecker  # Install the spellchecker library

# Install poppler-utils for pdf2image1
!sudo apt-get install poppler-utils

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import json
import pandas as pd
import cv2
import numpy as np
import re
import pytesseract
from pdf2image import convert_from_path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
import seaborn as sns
import easyocr
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm

# Set the Tesseract path (if needed)
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

# Dataset Path
data_dir = '/content/drive/MyDrive/OCR'
train_images_dir = os.path.join(data_dir, 'Training', 'training_words')
train_labels_file = os.path.join(data_dir, 'Training', 'training_labels.csv')
val_images_dir = os.path.join(data_dir, 'Validation', 'validation_words')
val_labels_file = os.path.join(data_dir, 'Validation', 'validation_labels.csv')
test_images_dir = os.path.join(data_dir, 'Testing', 'testing_words')
test_labels_file = os.path.join(data_dir, 'Testing', 'testing_labels.csv')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# Function to validate medicine names using regex
def validate_medicine_name(name):
    pattern = r'^[A-Za-z0-9\s\-]+$'
    return re.match(pattern, name) is not None

# Function to correct spelling in OCR text
def correct_spelling(text):
    if not isinstance(text, str) or not text.strip():
        return text
    spell = SpellChecker()
    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = word
        corrected_words.append(corrected_word)
    corrected_text = " ".join(corrected_words)
    return corrected_text

# Custom Dataset Class
class MedicalPrescriptionDataset(Dataset):
    def __init__(self, image_folder, label_file, feature_extractor, label_encoder=None):
        self.image_folder = image_folder
        self.label_file = label_file
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

        # Load labels
        self.labels_df = pd.read_csv(label_file)
        self.image_paths = []
        self.labels = []
        for _, row in self.labels_df.iterrows():
            img_name = row['IMAGE']
            img_path = os.path.join(image_folder, img_name)
            if os.path.exists(img_path):
                medicine_name = row['MEDICINE_NAME']
                if validate_medicine_name(medicine_name):
                    self.image_paths.append(img_path)
                    self.labels.append(medicine_name)

        # Encode labels if label_encoder is provided
        if self.label_encoder is not None:
            self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)
        encoding = self.feature_extractor(image, return_tensors="pt")
        label = self.labels[idx]
        return encoding['pixel_values'].squeeze(), label

# Load datasets
def load_datasets(train_images_dir, train_labels_file, val_images_dir, val_labels_file, test_images_dir, test_labels_file):
    # Initialize feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    # Initialize label encoder
    label_encoder = LabelEncoder()

    # Load training data
    train_dataset = MedicalPrescriptionDataset(train_images_dir, train_labels_file, feature_extractor)

    # Fit the label encoder on the training labels
    label_encoder.fit(train_dataset.labels)

    # Transform labels for all datasets
    train_dataset = MedicalPrescriptionDataset(train_images_dir, train_labels_file, feature_extractor, label_encoder)
    val_dataset = MedicalPrescriptionDataset(val_images_dir, val_labels_file, feature_extractor, label_encoder)
    test_dataset = MedicalPrescriptionDataset(test_images_dir, test_labels_file, feature_extractor, label_encoder)

    return train_dataset, val_dataset, test_dataset, label_encoder

# Load datasets
train_dataset, val_dataset, test_dataset, label_encoder = load_datasets(
    train_images_dir, train_labels_file, val_images_dir, val_labels_file, test_images_dir, test_labels_file
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(label_encoder.classes_),
    ignore_mismatched_sizes=True
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop with progress bars
def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training phase
        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()

        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': (predicted == labels).sum().item() / labels.size(0)
                })

        # Validation phase
        val_loss = 0
        val_correct = 0
        val_total = 0
        model.eval()

        with torch.no_grad(), tqdm(val_loader, desc="Validating") as pbar:
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': (predicted == labels).sum().item() / labels.size(0)
                })

        # Print epoch statistics
        print(f"\nTraining Loss: {train_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {train_correct / train_total * 100:.2f}%")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_correct / val_total * 100:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, optimizer, epochs=3)

# Evaluate the model
def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'acc': (predicted == labels).sum().item() / labels.size(0)
            })

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluate on test set
evaluate_model(model, test_loader, label_encoder)

# Function to convert PDF to image
def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Function to extract specific data fields using regex
def extract_data_using_regex(text, pattern):
    matches = re.findall(pattern, text)
    return matches

# Function to convert image from RGB to grayscale
def rgb_to_grayscale(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# Function to apply adaptive thresholding
def apply_adaptive_threshold(image):
    gray_image = rgb_to_grayscale(image)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Function to organize and store extracted data in JSON format
def store_data_in_json(data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage of the above functions
pdf_path = '/content/drive/MyDrive/OCR/sample.pdf'

# Verify that the file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist. Please upload the file to Google Drive.")

# Convert PDF to images
images = pdf_to_image(pdf_path)

# Process each image
extracted_data = []
for i, image in enumerate(images):
    # Convert to grayscale
    gray_image = rgb_to_grayscale(image)

    # Apply adaptive thresholding
    thresholded_image = apply_adaptive_threshold(image)

    # Perform OCR using EasyOCR
    result = reader.readtext(np.array(thresholded_image))
    text = " ".join([res[1] for res in result])

    # Extract specific data fields using regex (example: extract dates)
    date_pattern = r'\d{2}/\d{2}/\d{4}'
    dates = extract_data_using_regex(text, date_pattern)

    # Store extracted data
    extracted_data.append({
        'image_index': i,
        'text': text,
        'dates': dates
    })

# Store extracted data in JSON format
json_path = '/content/drive/MyDrive/OCR/extracted_data.json'
store_data_in_json(extracted_data, json_path)
print(f"\nExtracted data stored in {json_path}")
