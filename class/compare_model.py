import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'Decision Tree' : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost'       : XGBClassifier(n_estimators=100, random_state=42,
                                    eval_metric='logloss')
}

print(f"{'Model':<20} {'CV Mean':>10} {'CV Std':>10} {'Test':>10}")
print("-" * 55)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    test_score = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<20} {cv_scores.mean():>10.4f} "
          f"{cv_scores.std():>10.4f} {test_score:>10.4f}")


# XGBoost can track performance per iteration
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.001,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

# Fit with evaluation set
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Plot learning curves
results  = model.evals_result()
train_loss = results['validation_0']['logloss']
test_loss  = results['validation_1']['logloss']

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss,  label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.show()