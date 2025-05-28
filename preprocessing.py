import numpy as np
import cv2
import random
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


# CLAHE (contrast limited adaptive histogram equalization)
def apply_clahe(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = apply_clahe(img)
    return img

def preprocess_dataset(images):
    return np.array([preprocess_image(img) for img in images])

# For SVM model
def preprocess_for_svm(X_train, y_train, X_test, y_test):

    X_train = preprocess_dataset(X_train)
    X_test = preprocess_dataset(X_test)

    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)

    return X_train, y_train, X_test, y_test

def normalize_images(images):
    return images.astype('float32') / 255.0

# For ResNet50 model
def preprocess_for_resnet(X_train, y_train, X_test, y_test):

    X_train = preprocess_dataset(X_train)
    X_test = preprocess_dataset(X_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = resnet_preprocess(X_train)
    X_test = resnet_preprocess(X_test)

    return X_train, y_train, X_test, y_test


