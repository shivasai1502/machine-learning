"""
NumPy Basics - Foundation for Machine Learning
================================================
NumPy is the fundamental package for numerical computing in Python.
Understanding NumPy is crucial for ML as most libraries build on top of it.
"""

import numpy as np

print("=" * 60)
print("1. CREATING ARRAYS")
print("=" * 60)

# Create arrays from lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

print(f"1D Array: {arr1d}")
print(f"2D Array:\n{arr2d}")
print(f"Shape of 2D array: {arr2d.shape}")
print(f"Data type: {arr2d.dtype}\n")

# Special arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
random_arr = np.random.rand(3, 3)

print(f"Zeros:\n{zeros}\n")
print(f"Ones:\n{ones}\n")
print(f"Identity:\n{identity}\n")
print(f"Random:\n{random_arr}\n")

print("=" * 60)
print("2. ARRAY OPERATIONS")
print("=" * 60)

# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")
print(f"Square root of b: {np.sqrt(b)}\n")

print("=" * 60)
print("3. INDEXING AND SLICING")
print("=" * 60)

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"Original array:\n{arr}\n")
print(f"Element at [1, 2]: {arr[1, 2]}")
print(f"First row: {arr[0, :]}")
print(f"Second column: {arr[:, 1]}")
print(f"Subarray:\n{arr[0:2, 1:3]}\n")

# Boolean indexing
print(f"Elements > 5:\n{arr[arr > 5]}\n")

print("=" * 60)
print("4. STATISTICAL OPERATIONS")
print("=" * 60)

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(f"Data: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard Deviation: {np.std(data)}")
print(f"Variance: {np.var(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Sum: {np.sum(data)}\n")

print("=" * 60)
print("5. MATRIX OPERATIONS (Important for ML!)")
print("=" * 60)

# Matrix multiplication
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

print(f"Matrix X:\n{X}\n")
print(f"Matrix Y:\n{Y}\n")
print(f"Matrix multiplication (X @ Y):\n{X @ Y}\n")
print(f"Transpose of X:\n{X.T}\n")

# Dot product (very common in ML)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"Dot product of {v1} and {v2}: {np.dot(v1, v2)}\n")

print("=" * 60)
print("6. BROADCASTING (Key ML Concept)")
print("=" * 60)

# Broadcasting allows operations on arrays of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
vector = np.array([10, 20, 30])

print(f"Matrix:\n{matrix}\n")
print(f"Vector: {vector}\n")
print(f"Matrix + Vector (broadcasting):\n{matrix + vector}\n")

print("=" * 60)
print("EXERCISE: Try These!")
print("=" * 60)
print("""
1. Create a 5x5 matrix of random numbers between 0 and 100
2. Find all elements greater than 50
3. Calculate the mean of each column
4. Normalize the matrix (subtract mean, divide by std dev)
""")
