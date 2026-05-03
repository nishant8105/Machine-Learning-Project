import pandas as pd
import numpy as np

# Using the California Housing dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# --- Create new features from existing ones ---

# Rooms per person (more meaningful than raw room count)
df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']

# Bedroom ratio (what fraction of rooms are bedrooms?)
df['bedroom_ratio'] = df['AveBedrms'] / df['AveRooms']

# Income per room (wealth density)
df['income_per_room'] = df['MedInc'] / df['AveRooms']

print(df[['AveRooms', 'AveOccup', 'rooms_per_person', 
          'bedroom_ratio', 'income_per_room']].head())