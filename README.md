ZooTag Deep Learning Project

A deep learning image classification model built to automatically identify animal groups using transfer learning with **MobileNetV2**.  
Developed and trained in **Google Colab** using **TensorFlow/Keras**, with a lightweight **TensorFlow Lite** version for mobile deployment.

Tools & Libraries

Python

TensorFlow / Keras

NumPy · Matplotlib · scikit-learn

Google Colab · OpenCV

TensorFlow Lite

---

## 📘 Overview

The **ZooTag Challenge** focuses on recognizing and classifying animal images into seven main categories:

> **Amphibians · Birds · Fish · Insects · Mammals · Marine Invertebrates · Reptiles**

The goal was to design a compact, accurate model that performs well on unseen data and can be deployed on mobile or edge devices.

---

## ⚙️ Project Workflow

1. **Data Preparation**
   - Images collected and stored in Google Drive (`Lab19_images/`)
   - Split automatically using `validation_split=0.2` → **80% train / 10% validation / 10% test**
   - Resized to `224x224` and normalized to `[0, 1]`

2. **Data Augmentation**
   - Random horizontal flips  
   - Random brightness, contrast, saturation, and hue  
   - Random cropping / zooming  
   - Light Gaussian noise  

3. **Model Architecture**
   - **Base model:** `MobileNetV2` pretrained on ImageNet  
   - **Added layers:**
     - Global Average Pooling  
     - Dropout (rate = 0.3)  
     - Dense output layer (`softmax`, 7 classes)

4. **Training Strategy**
   - **Stage 1:** Feature extraction (base frozen, train new head)  
   - **Stage 2:** Fine-tuning (unfreeze last 20 layers, lower LR = `1e-4`)  
   - **Loss:** Sparse Categorical Cross-Entropy  
   - **Optimizer:** Adam (`1e-3` → `1e-4`)  
   - **Callbacks:** EarlyStopping, ReduceLROnPlateau  

5. **Evaluation**
   - Accuracy and SCCE Loss (train, val, test)  
   - Confusion matrix and per-class metrics  
   - TFLite verification (model consistency)

6. **Deployment**
   - Converted model to **TensorFlow Lite** (quantized, ~12 MB)  
   - Ready for Android/iOS or edge devices

---

## 📊 Results

| Dataset | Accuracy | Loss (SCCE) |
|----------|-----------|-------------|
| Training | 0.897 | 0.227 |
| Validation | 0.885 | 0.25–0.33 |
| Test | **0.883** | **0.33** |


## 🧠 Model Summary

| Item | Description |
|------|--------------|
| Architecture | MobileNetV2 (transfer learning) |
| Input size | 224 × 224 × 3 |
| Loss function | Sparse Categorical Cross-Entropy |
| Optimizer | Adam |
| Regularization | Dropout (0.3), Early Stopping |
| Framework | TensorFlow / Keras |
| Deployment | TensorFlow Lite (~12 MB) |

