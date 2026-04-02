# Day 5 - Machine Learning Journey

## Overview
Focused on understanding Feature Scaling and its importance in machine learning. Implemented scaling techniques to normalize data and improve model performance.

## What I Did

### feature_scaling.py
- Learned why feature scaling is important for ML algorithms
- Applied feature scaling techniques on dataset
- Implemented **Standardization (Z-score scaling)**
- Implemented **Normalization (Min-Max scaling)**
- Used scaling before training models to ensure better performance

## Key Learnings
- Feature scaling ensures all features are on the same scale
- Prevents features with large values from dominating the model
- Important for distance-based algorithms (KNN, K-Means, Gradient Descent)
- Standardization centers data around mean = 0 and std = 1
- Normalization scales values between 0 and 1

## Scaling Techniques

### 🔹 Standardization (Z-score)

z = (x - μ) / σ

Where:
- x → input feature value  
- μ → mean of the feature  
- σ → standard deviation of the feature  

---

### 🔹 Normalization (Min-Max)

x' = (x - x<sub>min</sub>) / (x<sub>max</sub> - x<sub>min</sub>)

Where:
- x → input feature value  
- x<sub>min</sub> → minimum value of the feature  
- x<sub>max</sub> → maximum value of the feature  
## Next Steps
- Apply feature scaling with different ML algorithms
- Compare model performance with and without scaling
- Explore Robust Scaling for outlier-heavy data
- Use scaling in pipelines

---
**Date:** `Day 5`  
**Files:** `feature_scaling.py`