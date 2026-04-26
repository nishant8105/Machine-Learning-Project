import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, n_iteration=1000):
        self.weight = None
        self.bias = 0
        self.lr = learning_rate
        self.n_iter = n_iteration
        self.loss_history = []
    
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z) )
    
    def predict_proba(self, X):
        z = X @ self.weight + self.bias
        return self._sigmoid(z)
    
    def predict_class(self, X, thershold = 0.5):
        return (self.predict_proba(X) >= thershold).astype(int)
    
    def _loss(self, y_true, y_pred) :
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true)* np.log(1 - y_pred))

    def fit(self, X, y):
        n_sample, n_feature = X.shape 

        self.weight = np.zeros(n_feature)
        self.bias = 0.0
        for _ in range(self.n_iter):

            y_pred = self.predict_proba(X)

            dw = (1/n_sample) * X.T @ (y_pred - y)
            db = (1/n_sample) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

            loss = self._loss(y, y_pred)
            self.loss_history.append(loss)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=5,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(
    learning_rate = 0.1,
    n_iteration = 1000
)
model.fit(X_train, y_train)

preds = model.predict_class(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
# Compare your scratch vs sklearn
from sklearn.linear_model import LogisticRegression as SklearnLR

sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)
sklearn_preds = sklearn_model.predict(X_test)

print("Scratch accuracy :", accuracy_score(y_test, preds))
print("Sklearn accuracy :", accuracy_score(y_test, sklearn_preds))
print("Difference        :", abs(
    accuracy_score(y_test, preds) -
    accuracy_score(y_test, sklearn_preds)
))