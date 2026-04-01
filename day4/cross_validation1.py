from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 4, 5, 4])

# model
model = LinearRegression()

# cross validation
scores = cross_val_score(model, x, y, cv=3)

print("Scores:", scores)
print("Average Score:", scores.mean())