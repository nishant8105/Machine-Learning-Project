import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

baseline_df = df[['Pclass','Age','SibSp','Parch','Fare']].copy()
baseline_df = baseline_df.fillna(baseline_df.median())

y = df['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_score = cross_val_score(model, baseline_df, y, cv=5).mean()
print(f"Baseline score: {baseline_score:.4f}")

df['Family_size'] = df['SibSp'] + df['Parch'] + 1
df['Is_alone'] = (df['Family_size'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
common_titles = ["Mr", "Mrs", "Miss", "Master"]

df['Title'] = df['Title'].apply(
    lambda x: x if x in common_titles else "Other"
)

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Age_group'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 60, np.inf],
    labels=['Child', 'Teen', 'Adult', 'Senior']
)

df['Fare_per_person'] = df['Fare'] / df['Family_size']

feature_df = df[['Pclass','Age','SibSp','Parch','Fare',
                 'Is_alone','Family_size','Title',
                 'Age_group','Fare_per_person']].copy()

num_cols = feature_df.select_dtypes(include=np.number).columns
feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())

cat_cols = feature_df.select_dtypes(exclude=np.number).columns
feature_df[cat_cols] = feature_df[cat_cols].fillna("Missing")

feature_df = pd.get_dummies(feature_df, columns=['Title', 'Age_group'])


feature_score = cross_val_score(model, feature_df, y, cv=5).mean()
print(f"Feature score: {feature_score:.4f}")

# After fitting your model
model.fit(feature_df, y)

importance = pd.DataFrame({
    'feature'   : feature_df.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))