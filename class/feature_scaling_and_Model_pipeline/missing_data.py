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