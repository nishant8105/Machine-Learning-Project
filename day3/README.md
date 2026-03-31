# Day 3 - Machine Learning Journey

## Overview
Focused on implementing Linear Regression using Scikit-Learn, splitting data into training and test sets, and evaluating model performance with the R² score.

## What I Did

### linear_regression.py
- Prepared feature matrix `X` and target array `Y` using NumPy
- Reshaped input data for compatibility with sklearn
- Split dataset into training (80%) and test (20%) sets using `train_test_split`
- Trained a `LinearRegression` model on the training data
- Generated predictions on the test set
- Evaluated model performance using the R² score

## Key Learnings
- How to structure data for sklearn models (`reshape(-1, 1)` for single features)
- Role of `train_test_split` in preventing overfitting and enabling fair evaluation
- How `random_state` ensures reproducible splits
- Interpreting R² score — values near 1.0 indicate a strong linear fit
- Difference between training the model and evaluating it on unseen data

## Next Steps
- Visualize the regression line against actual data points
- Experiment with polynomial features for non-linear relationships
- Explore other regression metrics (MAE, RMSE)
- Try multiple features (multivariable regression)

---
**Date:** Day 3  
**Files:** `train_test.py`
