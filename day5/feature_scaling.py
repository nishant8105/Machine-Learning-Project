from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1000], [2000], [3000], [4000]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original:\n", X)
print("Scaled:\n", X_scaled)