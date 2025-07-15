# MedViT-OCR: Medical Prescription Recognition using Vision Transformer (ViT)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Colab](https://img.shields.io/badge/Run%20on-Colab-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 🧠 Project Title
*Medical Prescription OCR with Vision Transformer (ViT) and EasyOCR*

## 📌 Description
This project implements an AI-based Optical Character Recognition (OCR) system for medical prescriptions using Google's Vision Transformer (ViT) for medicine name classification and EasyOCR for text extraction. The system includes advanced preprocessing, spelling correction, and data validation to handle handwritten and printed medical prescriptions.

---

## 🗂 Folder Structure
/OCR
├── Training/
│ ├── training_words/ (word images)
│ └── training_labels.csv
├── Validation/
│ ├── validation_words/ (word images)
│ └── validation_labels.csv
├── Testing/
│ ├── testing_words/ (word images)
│ └── testing_labels.csv
└── sample.pdf (example prescription)


---

## 🚀 Key Features
- Vision Transformer (ViT) for medicine name classification
- EasyOCR integration for robust text extraction
- Advanced image preprocessing (grayscale conversion, adaptive thresholding)
- Automatic spelling correction for OCR outputs
- Regex-based medicine name validation
- PDF to image conversion for document processing
- Comprehensive evaluation metrics (accuracy, confusion matrix)
- Google Colab compatible with Drive integration

---

## 📦 Requirements

```bash
pip install torch torchvision transformers
pip install pytesseract pdf2image opencv-python
pip install scikit-learn easyocr pyspellchecker
sudo apt-get install poppler-utils
