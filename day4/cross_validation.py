import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 4, 5, 4])

k = 3
fold_size = len(x) // k

scores = []

for i in range(k):
    start = i * fold_size
    end = start + fold_size

    # test split
    x_test = x[start:end]
    y_test = y[start:end]

    # train split
    x_train = np.concatenate((x[:start], x[end:]))
    y_train = np.concatenate((y[:start], y[end:]))

    # simple model (mean-based prediction just for demo)
    y_pred = np.mean(y_train)

    # simple score (not real R2, just demo)
    error = np.mean((y_test - y_pred)**2)
    scores.append(error)

print("Scores:", scores)
print("Average Error:", np.mean(scores))