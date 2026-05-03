from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

# Create a sample dataset with missing values
data = {
    'bedrooms':  [3, 2, np.nan, 4, 3],
    'price':     [300, 200, 250, np.nan, 350],
    'location':  ['urban', 'rural', np.nan, 'urban', 'rural']
}
df = pd.DataFrame(data)

print(df.isnull().sum())   # Count missing values per column

# Option 1: Drop rows with any missing value
df_dropped = df.dropna()

# Option 2: Fill numbers with the median (safer than mean — outliers don't skew it)
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
df['price'].fillna(df['price'].median(), inplace=True)

# Option 3: Fill text/category with the most common value (mode)
df['location'].fillna(df['location'].mode()[0], inplace=True)

print(df)  # No more NaN!
df = pd.DataFrame(data)
X = df[['bedrooms', 'price']]

# StandardScaler: mean=0, std=1  (best for most models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler: squeezes everything between 0 and 1
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

print("Standard scaled:\n", X_scaled)
print("MinMax scaled:\n", X_minmax)