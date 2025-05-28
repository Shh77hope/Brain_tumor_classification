# Brain Tumor MRI Classification

## Overview
This project trains and evaluates machine learning and deep learning models to classify brain MRI scans into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Supported models:
- **SVM** (Support Vector Machine)
- **ResNet50** (Convolutional Neural Network with transfer learning)

Grad-CAM visualizations can also be generated for ResNet50 models.

---

## Setup
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
1. The Dataset structure should like this: Brain_tumor_dataset/
    training/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    testing/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    
2. Running the Code
SVM model:
python main.py --model svm
ResNet50 model:
python main.py --model resnet --gradcam

