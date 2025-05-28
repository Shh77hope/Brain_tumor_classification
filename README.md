# 🧠 Brain Tumor MRI Classification

## 📌 Overview
This project leverages both traditional machine learning and deep learning techniques to classify brain MRI scans into four tumor categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## ✅ Models Implemented:
- **Support Vector Machine (SVM)** with PCA for dimensionality reduction.
- **ResNet50** with transfer learning, fine-tuning, and Grad-CAM interpretability.

---

## 📁 Dataset
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Organize the dataset as follows:

Brain_tumor_dataset/

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
    
## Run the Code
 
To train and evaluate the SVM model:

python main.py --model svm

To train and evaluate the ResNet50 model with Grad-CAM:

python main.py --model resnet --gradcam


## 📊 Performance

- ResNet50 Accuracy: ~98%

- SVM Accuracy: ~84% (with PCA)

Model performance is visualized with:

- Training & validation accuracy/loss plots

- Confusion matrix

- Grad-CAM heatmaps (for deep learning model interpretability)



