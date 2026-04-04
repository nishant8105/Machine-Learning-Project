import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 4, 4, 5 ,4 ]

df = pd.DataFrame({
    'Week' : x,
    'Sales' : y
})
X = df['Week'].to_numpy()
Y = df['Sales'].to_numpy()

mean_x = np.mean(x)
mean_y = np.mean(y)

n = len(X)

numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

m = numer/denom
c = mean_y - (m * mean_x)

print(f"m : {m}")
print(f"c : {c.round(2)}")

max_x = np.max(x) + 1
min_x = np.min(x) - 1

x = np.linspace(min_x, max_x)
y = (m * X) + c

plt.scatter(X,Y, color = '#043078', label='Data point')
plt.plot(x, y, color='#8c250f', label='Regression Line', linestyle = '--')
plt.xlabel("Weeks")
plt.ylabel("Sales")
plt.title('LR Demo')
plt.grid(True, alpha = 0.4, linestyle = ':')
plt.legend(loc= 'best')
plt.show()
X = X.reshape((n, 1))
model = LinearRegression()
model.fit(X, Y)

Y_predict = (m * X) + c
r2 = model.score(X, Y)
print(f"R2 Score : {r2}")