# Cancer-Classifier
KNN cancer classifier built on the Breast Cancer Wisconsin dataset. Clean dataset exploration, simple model tuning (k=1–100), and a clear accuracy plot that tells you where the model performs best. A compact, no-nonsense intro to machine learning.

# Breast Cancer Classifier — KNN Model

A simple machine learning project that classifies breast cancer tumors as **malignant** or **benign** using a **k-Nearest Neighbors (KNN)** model. Built using the Breast Cancer Wisconsin dataset from scikit-learn.

---

## Overview
This project:
- Loads and inspects the dataset  
- Prints target labels and meanings  
- Splits data into training and validation sets  
- Trains a KNN classifier across `k = 1 to 100`  
- Tracks validation accuracy for each `k`  
- Plots the accuracy performance curve  

A clean introduction to supervised learning and parameter tuning.

---

## What the Script Does
### 1. Load the dataset
```python
breast_cancer_data = load_breast_cancer()
