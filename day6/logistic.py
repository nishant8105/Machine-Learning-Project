import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (log loss)
def compute_cost(X, y, weights, bias):
    m = len(y)
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    
    cost = (-1/m) * np.sum(
        y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
    )
    return cost

# Gradient descent
def train(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    
    # Initialize weights and bias
    weights = np.zeros(n)
    bias = 0
    
    for i in range(epochs):
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Gradients
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Print cost occasionally
        if i % 100 == 0:
            cost = compute_cost(X, y, weights, bias)
            print(f"Epoch {i}, Cost: {cost:.4f}")
    
    return weights, bias

# Prediction function
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return [1 if p >= 0.5 else 0 for p in probs]


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

# Train model
weights, bias = train(X, y, learning_rate=0.01, epochs=1000)

# Predict
preds = predict(X, weights, bias)

print("\nPredictions:", preds)
print("Actual     :", y.tolist())