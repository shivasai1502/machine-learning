"""
Decision Trees - Intuitive Classification & Regression
=======================================================
Decision trees make predictions by learning simple decision rules from data.
Think of it like a flowchart: "If X > 5 and Y < 3, then Class A"

Very interpretable and forms the basis for powerful ensemble methods!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from pathlib import Path

print("=" * 60)
print("DECISION TREES - IRIS FLOWER CLASSIFICATION")
print("=" * 60)

# Load the famous Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

print(f"Dataset: {len(X)} samples")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")
print(f"Samples per class: {np.bincount(y)}\n")

# For visualization, let's use only 2 features
X_2d = X[:, [2, 3]]  # petal length and width
feature_names_2d = [feature_names[2], feature_names[3]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

print("=" * 60)
print("COMPARING TREE DEPTHS")
print("=" * 60)

# Try different max depths
depths = [1, 2, 3, 5, None]  # None = unlimited depth
results = []

for depth in depths:
    # Create and train model
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    # Predict
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    depth_str = str(depth) if depth else "Unlimited"
    results.append({
        'depth': depth,
        'depth_str': depth_str,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_nodes': dt.tree_.node_count,
        'n_leaves': dt.tree_.n_leaves
    })

    print(f"Depth {depth_str:9s} → Train: {train_acc:.2%}, Test: {test_acc:.2%}, "
          f"Nodes: {dt.tree_.node_count:3d}, Leaves: {dt.tree_.n_leaves:3d}")

# Find best depth based on test accuracy
best = max(results, key=lambda x: x['test_acc'])
print(f"\nBest depth: {best['depth_str']} (Test accuracy: {best['test_acc']:.2%})")

# Train final model
print("\n" + "=" * 60)
print(f"FINAL MODEL (max_depth={best['depth']})")
print("=" * 60)

final_model = DecisionTreeClassifier(max_depth=best['depth'], random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
importances = final_model.feature_importances_
for feature, importance in zip(feature_names_2d, importances):
    print(f"{feature:25s}: {importance:.4f} {'█' * int(importance * 50)}")

# Visualize the tree structure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Plot 1: Shallow tree (depth=2)
ax = axes[0, 0]
shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
shallow_tree.fit(X_train, y_train)
plot_tree(shallow_tree, feature_names=feature_names_2d,
          class_names=class_names, filled=True, ax=ax, fontsize=10)
ax.set_title('Shallow Tree (depth=2) - Easy to Interpret', fontsize=14, fontweight='bold')

# Plot 2: Deep tree (unlimited)
ax = axes[0, 1]
deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree.fit(X_train, y_train)
plot_tree(deep_tree, feature_names=feature_names_2d,
          class_names=class_names, filled=True, ax=ax, fontsize=6)
ax.set_title('Deep Tree (unlimited) - May Overfit', fontsize=14, fontweight='bold')

# Plot 3: Decision boundary for best model
ax = axes[1, 0]
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = final_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                    cmap='viridis', edgecolors='black', s=100)
ax.set_xlabel(feature_names_2d[0], fontsize=12)
ax.set_ylabel(feature_names_2d[1], fontsize=12)
ax.set_title('Decision Boundary', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax)

# Plot 4: Depth vs Accuracy
ax = axes[1, 1]
depth_labels = [r['depth_str'] for r in results]
train_accs = [r['train_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]
x_pos = np.arange(len(depth_labels))

width = 0.35
ax.bar(x_pos - width/2, train_accs, width, label='Training', alpha=0.8)
ax.bar(x_pos + width/2, test_accs, width, label='Testing', alpha=0.8)
ax.set_xlabel('Max Depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(depth_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = Path(__file__).parent / 'decision_tree_results.png'
plt.savefig(output_path, dpi=150)
print(f"
Visualization saved to: {output_path}")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. HOW IT WORKS:
   - Splits data recursively based on feature values
   - Chooses splits that best separate classes
   - Creates rectangular decision regions

2. SPLITTING CRITERIA:
   - Gini Impurity: Measures class mixture (default for classification)
   - Entropy: Information gain (alternative measure)
   - MSE: For regression trees

3. HYPERPARAMETERS:
   - max_depth: Limits tree depth (prevents overfitting)
   - min_samples_split: Minimum samples to split a node
   - min_samples_leaf: Minimum samples in leaf node
   - max_features: Features to consider for best split

4. PROS:
   ✓ Highly interpretable (can visualize decisions)
   ✓ Handles non-linear relationships
   ✓ No feature scaling needed
   ✓ Handles mixed data types
   ✓ Feature importance built-in

5. CONS:
   ✗ Prone to overfitting (especially deep trees)
   ✗ Unstable (small data changes → different tree)
   ✗ Biased toward dominant classes
   ✗ Not great for extrapolation

6. OVERFITTING VS UNDERFITTING:
   - Shallow tree (low depth) → Underfitting → High bias
   - Deep tree (high/no depth) → Overfitting → High variance
   - Use validation to find sweet spot!

TIP: Decision trees are rarely used alone in practice.
     They shine in ensemble methods like Random Forest and Gradient Boosting!

EXERCISE:
- Try different splitting criteria (gini vs entropy)
- Experiment with min_samples_split and min_samples_leaf
- Use all 4 features instead of just 2
- Build a regression tree for continuous targets
""")
