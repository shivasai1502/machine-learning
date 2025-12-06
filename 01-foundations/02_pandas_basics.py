"""
Pandas Basics - Data Manipulation for Machine Learning
=======================================================
Pandas is essential for data preprocessing and exploration.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("1. CREATING DATAFRAMES")
print("=" * 60)

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Score': [85.5, 92.0, 78.5, 88.0, 95.5],
    'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}\n")

print("=" * 60)
print("2. BASIC OPERATIONS")
print("=" * 60)

print("First 3 rows:")
print(df.head(3))
print("\nBasic statistics:")
print(df.describe())
print("\nColumn names:", df.columns.tolist())
print("\nInfo:")
print(df.info())

print("\n" + "=" * 60)
print("3. SELECTING DATA")
print("=" * 60)

# Select column
print("Ages:")
print(df['Age'])

# Select multiple columns
print("\nNames and Scores:")
print(df[['Name', 'Score']])

# Select rows by condition
print("\nPeople older than 28:")
print(df[df['Age'] > 28])

# Multiple conditions
print("\nPeople older than 28 with score > 90:")
print(df[(df['Age'] > 28) & (df['Score'] > 90)])

print("\n" + "=" * 60)
print("4. DATA MANIPULATION")
print("=" * 60)

# Add new column
df['Score_Normalized'] = (df['Score'] - df['Score'].mean()) / df['Score'].std()
print("With normalized score:")
print(df)

# Group by operations
print("\nAverage age by city:")
print(df.groupby('City')['Age'].mean())

print("\nStatistics by city:")
print(df.groupby('City').agg({
    'Age': 'mean',
    'Score': ['mean', 'max', 'min']
}))

print("\n" + "=" * 60)
print("5. HANDLING MISSING DATA")
print("=" * 60)

# Create data with missing values
data_missing = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, np.nan, 8, 9],
    'C': [10, 11, 12, 13, 14]
}

df_missing = pd.DataFrame(data_missing)
print("Data with missing values:")
print(df_missing)

print("\nMissing values count:")
print(df_missing.isnull().sum())

print("\nFill missing with mean:")
df_filled = df_missing.fillna(df_missing.mean())
print(df_filled)

print("\nDrop rows with missing values:")
df_dropped = df_missing.dropna()
print(df_dropped)

print("\n" + "=" * 60)
print("6. SORTING AND RANKING")
print("=" * 60)

print("Sort by Score (descending):")
print(df.sort_values('Score', ascending=False))

print("\nRank scores:")
df['Rank'] = df['Score'].rank(ascending=False)
print(df[['Name', 'Score', 'Rank']])

print("\n" + "=" * 60)
print("EXERCISE: Try These!")
print("=" * 60)
print("""
1. Create a DataFrame with 100 rows of random data
2. Filter rows based on multiple conditions
3. Group by a categorical column and calculate statistics
4. Handle missing values using different strategies
5. Create new features by combining existing columns
""")
