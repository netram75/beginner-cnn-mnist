# Beginner CNN: MNIST Digit Recognition

This project demonstrates handwritten digit classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained and evaluated on the MNIST dataset and achieves high classification accuracy.

---

## 📌 Project Overview

The objective of this project is to understand how Convolutional Neural Networks extract spatial features from image data and perform multi-class classification.

The model classifies grayscale images of handwritten digits (0–9).

---

## 📊 Dataset

- Dataset: MNIST
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 pixels
- Channels: 1 (Grayscale)
- Classes: 10 (Digits 0–9)

The dataset is loaded using TensorFlow’s built-in dataset loader.

---

## 🏗 Model Architecture

The CNN architecture includes:

- Input Layer
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Flatten Layer
- Dense Layer (128 neurons)
- Output Layer (Softmax - 10 classes)

Optimizer: Adam  
Loss Function: Categorical Crossentropy  

---

## 📈 Results

- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Test Accuracy: **98.8%**

The model generalizes well with minimal overfitting.

---

## 🔍 Error Analysis

Confusion matrix analysis shows minor confusion between visually similar digits such as:
- 5 and 3
- 9 and 4
- 6 and 0

This behavior is expected due to handwriting variations.

---

## 🚀 Kaggle Notebook

You can view the full training notebook here:

🔗 https://www.kaggle.com/code/netramfaran/beginner-cnn-mnist-digit-recognition

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

---

## 🎯 Key Learnings

- CNN feature extraction
- Multi-class classification
- Model evaluation and confusion matrix analysis
- Proper preprocessing techniques for image data

---
