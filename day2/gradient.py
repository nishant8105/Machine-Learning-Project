import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 4, 5, 4]

df = pd.DataFrame({
    'Week': x,
    'Sales': y
})

X = df['Week'].to_numpy()
Y = df['Sales'].to_numpy()

n = len(X)

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
iterations = 1000

# Gradient Descent
for i in range(iterations):
    y_pred = m * X + b
    
    dm = (-2/n) * np.sum(X * (Y - y_pred))
    db = (-2/n) * np.sum(Y - y_pred)
    
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Results
print(f"Gradient Descent m: {m:.4f}")
print(f"Gradient Descent b: {b:.4f}")

# Plot
max_x = np.max(X) + 1
min_x = np.min(X) - 1

x_line = np.linspace(min_x, max_x)
y_line = m * x_line + b

plt.scatter(X, Y, label='Data point')
plt.plot(x_line, y_line, label='GD Regression Line', linestyle='--')
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.title('Gradient Descent LR')
plt.grid(True)
plt.legend()
plt.show()

# Compare with sklearn
X_reshaped = X.reshape((n, 1))
model = LinearRegression()
model.fit(X_reshaped, Y)

print(f"Sklearn m: {model.coef_[0]:.4f}")
print(f"Sklearn b: {model.intercept_:.4f}")
