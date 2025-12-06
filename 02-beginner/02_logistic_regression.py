"""
Logistic Regression - Binary Classification
============================================
Despite the name, logistic regression is used for CLASSIFICATION, not regression.
It predicts probabilities and classifies data into categories (e.g., spam/not spam).

Output: Probability between 0 and 1 using sigmoid function
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

print("=" * 60)
print("LOGISTIC REGRESSION - EMAIL SPAM DETECTION")
print("=" * 60)

# Generate synthetic dataset: Email features vs Spam/Not Spam
np.random.seed(42)
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.1,  # Add 10% noise
    random_state=42
)

print(f"Dataset: {len(X)} emails")
print(f"Features: Word count, Link count (simplified)")
print(f"Classes: 0 = Not Spam, 1 = Spam")
print(f"Spam emails: {np.sum(y == 1)}")
print(f"Normal emails: {np.sum(y == 0)}\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} emails")
print(f"Test set: {len(X_test)} emails\n")

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)

print("\nExample predictions with probabilities:")
for i in range(5):
    print(f"Email {i+1}: Class {y_pred_test[i]} "
          f"(Not Spam: {y_pred_proba[i][0]:.2%}, Spam: {y_pred_proba[i][1]:.2%})")

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"  True Negatives (correct 'not spam'): {cm[0][0]}")
print(f"  False Positives (wrongly marked spam): {cm[0][1]}")
print(f"  False Negatives (missed spam): {cm[1][0]}")
print(f"  True Positives (correct 'spam'): {cm[1][1]}\n")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_test,
                          target_names=['Not Spam', 'Spam']))

# Visualize decision boundary
plt.figure(figsize=(15, 5))

# Plot 1: Training data
plt.subplot(1, 3, 1)
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                     cmap='coolwarm', alpha=0.6, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)

# Plot 2: Test data with predictions
plt.subplot(1, 3, 2)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test,
                     cmap='coolwarm', alpha=0.6, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data - Predictions')
plt.colorbar(scatter, label='Predicted Class')
plt.grid(True, alpha=0.3)

# Plot 3: Decision boundary
plt.subplot(1, 3, 3)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
           cmap='coolwarm', alpha=0.8, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machine-learning/02-beginner/logistic_regression_results.png', dpi=150)
print("\nVisualization saved to: 02-beginner/logistic_regression_results.png")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. SIGMOID FUNCTION: Converts any value to probability (0 to 1)
   σ(x) = 1 / (1 + e^(-x))

2. DECISION BOUNDARY: Line/surface separating classes

3. METRICS:
   - Accuracy: Overall correctness
   - Precision: Of predicted spam, how many are actually spam?
   - Recall: Of actual spam, how many did we catch?
   - F1-Score: Balance between precision and recall

4. CONFUSION MATRIX:
                    Predicted
                 Not Spam  Spam
   Actual Not Spam   TN     FP
          Spam       FN     TP

WHEN TO USE:
✓ Binary classification (yes/no, spam/ham)
✓ Need probability estimates
✓ Linearly separable data
✗ Multi-class (use variants or other algorithms)
✗ Non-linear boundaries (try polynomial features or other models)

EXERCISE:
- Adjust the decision threshold (default 0.5)
- Try with imbalanced datasets
- Experiment with regularization (C parameter)
""")
