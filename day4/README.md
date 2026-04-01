# Day 4 - Machine Learning Journey

## Overview
Focused on understanding and implementing Cross Validation to evaluate model performance more reliably using multiple data splits.

## What I Did

### cross_validation.py
- Learned the concept of K-Fold Cross Validation
- Used dataset multiple times for training and testing
- Implemented cross validation using `cross_val_score` from sklearn
- Trained a `LinearRegression` model across multiple folds
- Calculated performance scores for each fold
- Computed the average score to evaluate overall model performance

## Key Learnings
- Cross Validation provides more reliable results than a single train-test split
- K-Fold splits data into multiple parts and rotates test sets
- Each data point gets a chance to be in the test set
- Helps reduce overfitting and improves model generalization
- Average score gives better estimate of real-world performance

## Next Steps
- Try different values of K (5, 10) and compare results
- Implement Stratified K-Fold for classification problems
- Combine Cross Validation with Hyperparameter Tuning
- Visualize performance across folds

---
**Date:** Day 4  
**Files:** `cross_validation.py`