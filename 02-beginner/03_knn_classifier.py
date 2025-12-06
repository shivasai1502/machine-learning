"""
K-Nearest Neighbors (KNN) - Instance-Based Learning
====================================================
KNN classifies a data point based on how its neighbors are classified.
It's simple, intuitive, and doesn't require training!

Key Idea: "You are the average of your k closest friends"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_moons

print("=" * 60)
print("K-NEAREST NEIGHBORS - PATTERN RECOGNITION")
print("=" * 60)

# Generate non-linear dataset (moons)
np.random.seed(42)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

print(f"Dataset: {len(X)} samples")
print(f"Classes: 2 (Moon 1, Moon 2)")
print(f"Features: 2D coordinates\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

print("=" * 60)
print("COMPARING DIFFERENT K VALUES")
print("=" * 60)

# Try different k values
k_values = [1, 3, 5, 10, 20]
results = []

for k in k_values:
    # Create and train model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    results.append({
        'k': k,
        'train_acc': train_acc,
        'test_acc': test_acc
    })

    print(f"k={k:2d} → Train: {train_acc:.2%}, Test: {test_acc:.2%}")

# Find best k
best_k = max(results, key=lambda x: x['test_acc'])
print(f"\nBest k value: {best_k['k']} (Test accuracy: {best_k['test_acc']:.2%})")

# Train final model with best k
print("\n" + "=" * 60)
print(f"TRAINING FINAL MODEL (k={best_k['k']})")
print("=" * 60)

final_model = KNeighborsClassifier(n_neighbors=best_k['k'])
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# Demonstrate prediction for a new point
new_point = np.array([[0.5, 0.5]])
prediction = final_model.predict(new_point)
distances, indices = final_model.kneighbors(new_point)

print("\n" + "=" * 60)
print("PREDICTION EXAMPLE")
print("=" * 60)
print(f"New point: {new_point[0]}")
print(f"Predicted class: {prediction[0]}")
print(f"\nNearest {best_k['k']} neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor = X_train[idx]
    neighbor_class = y_train[idx]
    print(f"  {i+1}. Distance: {dist:.3f}, Class: {neighbor_class}, Point: {neighbor}")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    ax = axes[idx]

    # Train model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Create decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
              cmap='coolwarm', edgecolors='black', s=50, alpha=0.8)

    test_acc = results[idx]['test_acc']
    ax.set_title(f'k={k} (Accuracy: {test_acc:.2%})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

# Last plot: k vs accuracy
ax = axes[5]
k_vals = [r['k'] for r in results]
train_accs = [r['train_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]

ax.plot(k_vals, train_accs, marker='o', label='Training', linewidth=2)
ax.plot(k_vals, test_accs, marker='s', label='Testing', linewidth=2)
ax.set_xlabel('k (number of neighbors)')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance vs k')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/machine-learning/02-beginner/knn_results.png', dpi=150)
print("\nVisualization saved to: 02-beginner/knn_results.png")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. DISTANCE METRICS:
   - Euclidean: √((x₁-x₂)² + (y₁-y₂)²)  [Most common]
   - Manhattan: |x₁-x₂| + |y₁-y₂|
   - Minkowski: Generalization of above

2. CHOOSING K:
   - Too small (k=1): Sensitive to noise, overfits
   - Too large (k=N): Underfits, ignores local patterns
   - Rule of thumb: k = √N (then tune with validation)
   - Always use odd k for binary classification (breaks ties)

3. PROS:
   ✓ Simple and intuitive
   ✓ No training phase (lazy learning)
   ✓ Works well with non-linear boundaries
   ✓ Naturally handles multi-class

4. CONS:
   ✗ Slow prediction (must compare to all training points)
   ✗ Memory intensive (stores all training data)
   ✗ Sensitive to feature scaling (normalize data!)
   ✗ Curse of dimensionality (struggles with many features)

5. IMPORTANT: Always normalize/standardize features before using KNN!

EXERCISE:
- Try different distance metrics
- Test with normalized vs unnormalized data
- Experiment with weighted KNN (closer neighbors have more influence)
- Apply to iris dataset (3 classes)
""")
