from decision_tree import DecisionTreeScratch
import numpy as np

class RandomForestScratch :

    def __init__(self, n_trees = 100, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)

        return X[indices], y[indices]
    
    def _get_feature_subset(self, n_features):
        if self.max_features == 'sqrt':
            size = int(np.sqrt(n_features))
        else : 
            size = n_features
        
        return np.random.choice(n_features, size=size, replace=False)
    
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            feature_indices = self._get_feature_subset(n_features)

            X_sample_subset = X_sample[:, feature_indices]

            tree = DecisionTreeScratch(max_depth=self.max_depth)
            tree.fit(X=X_sample_subset, y=y_sample)

            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        tree_preds = []

        for tree, feature_indices in self.trees:
            preds = tree.predict(X[:, feature_indices])
            tree_preds.append(preds)

        # shape → (n_trees, n_samples)
        tree_preds = np.array(tree_preds)

        # majority vote
        def majority_vote(x):
            return np.bincount(x).argmax()

        return np.apply_along_axis(majority_vote, axis=0, arr=tree_preds)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=500, n_features=10,
                              random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf = RandomForestScratch(n_trees=50, max_depth=5)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    print("Scratch RF :", accuracy_score(y_test, preds))

    sk_rf = RandomForestClassifier(n_estimators=50,
                                   max_depth=5, random_state=42)
    sk_rf.fit(X_train, y_train)

    print("Sklearn RF :", accuracy_score(y_test, sk_rf.predict(X_test)))