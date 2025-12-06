"""
Linear Regression - Your First ML Algorithm
============================================
Linear regression finds the best-fitting straight line through data points.
It's used for predicting continuous values.

Equation: y = mx + b (or y = w1*x1 + w2*x2 + ... + b)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

print("=" * 60)
print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 60)

# Generate synthetic data: House size vs Price
np.random.seed(42)
house_size = np.random.randint(500, 3500, 100)  # Square feet
# Price formula: $50k base + $150 per sqft + some noise
house_price = 50000 + 150 * house_size + np.random.normal(0, 25000, 100)

# Reshape for sklearn (needs 2D array)
X = house_size.reshape(-1, 1)
y = house_price

print(f"Dataset: {len(X)} houses")
print(f"House sizes range: {house_size.min()} - {house_size.max()} sqft")
print(f"Prices range: ${house_price.min():,.0f} - ${house_price.max():,.0f}\n")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} houses")
print(f"Test set: {len(X_test)} houses\n")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)
print(f"Slope (coefficient): ${model.coef_[0]:,.2f} per sqft")
print(f"Intercept: ${model.intercept_:,.2f}")
print(f"\nEquation: Price = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Size\n")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Training MSE: ${train_mse:,.2f}")
print(f"Test MSE: ${test_mse:,.2f}")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print("\nR² Score interpretation:")
print("  1.0 = Perfect prediction")
print("  0.8-0.9 = Very good")
print("  0.6-0.8 = Good")
print("  < 0.6 = Needs improvement\n")

# Make predictions on new data
new_houses = np.array([[1000], [2000], [3000]])
predictions = model.predict(new_houses)

print("=" * 60)
print("PREDICTIONS ON NEW DATA")
print("=" * 60)
for size, price in zip(new_houses, predictions):
    print(f"House size: {size[0]:,} sqft → Predicted price: ${price:,.2f}")

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Actual prices')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Fitted line')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Training Data - Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test data
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Actual prices', color='green')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Predictions')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Test Data - Model Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent / 'linear_regression_results.png'
plt.savefig(output_path, dpi=150)
print(f"
Visualization saved to: {output_path}")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. Training: Learning the relationship from data
2. Testing: Evaluating on unseen data
3. MSE: Average squared difference between actual and predicted
4. R² Score: How well the model explains variance (0 to 1)
5. Overfitting: Model too complex, memorizes training data
6. Underfitting: Model too simple, misses patterns

EXERCISE:
- Try with different train/test splits
- Add more features (multi-variate regression)
- Experiment with polynomial features
""")
