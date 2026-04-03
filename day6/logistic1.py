# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (you can replace this with your own)
# Features: [hours studied, hours slept]
X = np.array([
    [2, 5],
    [4, 6],
    [6, 7],
    [8, 8],
    [10, 9],
    [1, 4],
    [3, 5],
    [7, 7]
])

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 1, 1, 0, 0, 1])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Example: Predict if a student passes
new_data = np.array([[5, 6]])  # hours studied, hours slept
prediction = model.predict(new_data)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")