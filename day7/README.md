# Day 7 - Machine Learning Journey

## Overview
Focused on understanding Classification Metrics to evaluate the performance of classification models. Learned how to measure model accuracy beyond just correct predictions.

## What I Did

### logistic.py
- Learned different evaluation metrics for classification models
- Understood Confusion Matrix and its components
- Calculated Accuracy, Precision, Recall, and F1 Score
- Interpreted model performance using different metrics
- Understood when to use each metric

## Key Learnings
- Accuracy alone is not always reliable (especially for imbalanced data)
- Confusion Matrix gives detailed insight into predictions
- Precision measures correctness of positive predictions
- Recall measures how many actual positives were captured
- F1 Score balances Precision and Recall

## 📊 Confusion Matrix Terms

- True Positive (TP) → Correctly predicted positive  
- True Negative (TN) → Correctly predicted negative  
- False Positive (FP) → Incorrectly predicted positive  
- False Negative (FN) → Incorrectly predicted negative  

## 📐 Metrics Formulas

### 🔹 Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)

### 🔹 Precision
Precision = TP / (TP + FP)

### 🔹 Recall
Recall = TP / (TP + FN)

### 🔹 F1 Score
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

## Next Steps
- Implement confusion matrix in code using sklearn
- Visualize confusion matrix using heatmaps
- Learn ROC Curve and AUC Score
- Apply metrics on real-world datasets

---
**Date:** Day 7  
**Files:** `classification_metrics.py`