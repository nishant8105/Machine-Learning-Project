from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train a quick Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importances — how much does each feature help?
importances = pd.Series(rf.feature_importances_, index=data.feature_names)
importances = importances.sort_values(ascending=True)

# Plot
importances.plot(kind='barh', figsize=(8, 5), color='steelblue')
plt.title("Feature Importance — California Housing")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# Drop features with very low importance (e.g. below 0.05)
important_features = importances[importances > 0.05].index.tolist()
X_reduced = X[important_features]
print("Kept features:", important_features)