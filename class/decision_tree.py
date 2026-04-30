import numpy as np
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value 

class DecisionTreeScratch:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root      = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p**2)

    def best_split(self, X, y):
        best_feature = None
        best_thresh = None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[ :, feature])

            for thresh in thresholds:
                left_idx = X[:, feature] <= thresh
                right_idx = X[:, feature] > thresh

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0 :
                    continue
            
                y_left, y_right = y[left_idx], y[right_idx]

                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)

                weighted_gini = (
                    len(y_left) / n_samples * gini_left +
                    len(y_right) / n_samples * gini_right
                )

                if weighted_gini < best_gini :
                    best_gini = weighted_gini
                    best_feature = feature
                    best_thresh = thresh
        return best_feature, best_thresh

    def build_tree(self, X, y, depth=0):
        if len(y) == 0:
            return Node(value=0)

        if len(np.unique(y)) == 1 :
            return Node(value = y[0])

        if depth >= self.max_depth:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        feature, thresh = self.best_split(X, y)

        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        left_idx = X[:, feature] <= thresh
        right_idx = X[:, feature] > thresh

        left_subtree  = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=thresh, left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else : 
            return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=500,
    n_features=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tree = DecisionTreeScratch(max_depth=5)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Compare with sklearn
from sklearn.tree import DecisionTreeClassifier
sk_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)
print("Sklearn :", accuracy_score(y_test, sk_tree.predict(X_test)))