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
```

## 2. Inspect dataset components
- Prints target values
- Prints target names (malignant, benign)
- Prints all feature values
- Shows the label of the first data point

### 3. Train/ validation split:
```python
train_test_split(... test_size=0.2)
```

### 4. KNN loop (k = 1 → 100):
```python
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))
```

### 5. Plot results
- A matplotlib graph of accuracy vs. number of neighbors.

### 6. Output
The script generates a plot:
- X-axis: k-values
- Y-axis: validation accuracy
- Title: "KNN Accuracy vs Number of Neighbors"
- This helps identify the most effective k.

### 6. Requirements
Install the necessary libraries:
```python
pip install scikit-learn matplotlib
```
If you're not using Codecademy, you can remove the codecademylib3_seaborn import.

### 7. How to Run:
```python
python cancer_classifier.py
```

### 8. Dataset
Dataset: Breast Cancer Wisconsin (Diagnostic)
Available directly within scikit-learn.

Labels:
0 = malignant
1 = benign

### 9. Author
Thato Seluku
Power Platform Developer • ML Explorer • Builder of smarter tools
