# 🤞 Credit Card Fraud Detection

This project implements a robust machine learning pipeline for detecting fraudulent credit card transactions. It addresses the significant class imbalance problem using SMOTE (Synthetic Minority Over-sampling Technique) and provides an interactive Streamlit dashboard for monitoring and predictions.

## 📁 Project Structure

- `app/`: Streamlit dashboard and UI pages.
- `src/`: Core logic for data loading, preprocessing, training, and evaluation.
- `notebook/`: Exploratory Data Analysis and model experimentation.
- `data/`: Data storage (raw and processed).
- `models/`: Saved model binaries (`.pkl`).
- `tests/`: Automated tests for the pipeline.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (`venv`)

### Installation
1. Activate your virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit dashboard:
```bash
streamlit run app/main.py
```

## 🤖 Model Pipeline
The system compares three main models:
1. **Logistic Regression**: Baseline linear classifier.
2. **Random Forest**: Robust ensemble model.
3. **Gradient Boosting**: High-performance boosting classifier.

Hyperparameter tuning is handled via **Optuna** to maximize the F1-score, which is critical for fraud detection where minimizing false negatives is key.

## 📊 Dataset
The project uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. Ensure `creditcard.csv` is placed in `data/raw/` before training.
