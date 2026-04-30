import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate dataset
X, y = make_regression(n_samples=400, n_features=10,
                       noise=20, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    'No Regularization' : LinearRegression(),
    'L2 Ridge (λ=1)'    : Ridge(alpha=1),
    'L2 Ridge (λ=100)'  : Ridge(alpha=100),
    'L1 Lasso (λ=1)'    : Lasso(alpha=1),
    'L1 Lasso (λ=100)'  : Lasso(alpha=100),
}

print(f"{'Model':<25} {'Weights':<50}")
print("-" * 75)

for name, model in models.items():
    model.fit(X_train, y_train)
    weights = np.round(model.coef_, 2)
    print(f"{name:<25} {str(weights):<50}")