import pandas as pd

# NOTE: this dataset is semicolon-separated!
df = pd.read_csv("data/bank-full.csv", sep=";")

print("Shape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nTarget distribution (did they subscribe?):")
print(df["y"].value_counts())
print("\nAs percentages:")
print(df["y"].value_counts(normalize=True).round(3))
print("\nMissing values per column:\n", df.isnull().sum())