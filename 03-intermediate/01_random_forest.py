"""
Random Forest - Ensemble of Decision Trees
===========================================
Random Forest combines multiple decision trees to create a more robust model.
It's one of the most popular and powerful ML algorithms!

Key Idea: "Wisdom of the crowd" - many weak learners → strong learner
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
import seaborn as sns

print("=" * 60)
print("RANDOM FOREST - CREDIT RISK CLASSIFICATION")
print("=" * 60)

# Generate synthetic credit data
np.random.seed(42)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    flip_y=0.1,
    random_state=42
)

feature_names = [f'Feature_{i+1}' for i in range(10)]
print(f"Dataset: {len(X)} loan applications")
print(f"Features: {len(feature_names)} (income, credit score, debt, etc.)")
print(f"Classes: 0 = Low Risk, 1 = High Risk")
print(f"Class distribution: {np.bincount(y)}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

print("=" * 60)
print("COMPARING: SINGLE TREE VS RANDOM FOREST")
print("=" * 60)

# Single Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Random Forest with different numbers of trees
n_estimators_list = [1, 5, 10, 50, 100, 200]
rf_results = []

for n_est in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    rf_results.append({
        'n_estimators': n_est,
        'accuracy': rf_acc
    })

    print(f"Random Forest (n_estimators={n_est:3d}) → Accuracy: {rf_acc:.2%}")

print(f"\nSingle Decision Tree → Accuracy: {dt_acc:.2%}")

# Train final Random Forest
print("\n" + "=" * 60)
print("FINAL RANDOM FOREST MODEL (100 trees)")
print("=" * 60)

rf_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)
y_pred_proba = rf_final.predict_proba(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Low Risk', 'High Risk']))

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(rf_final, X_train, y_train, cv=5)
print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Get feature importances
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1:2d}. {feature_names[idx]:12s}: {importances[idx]:.4f} "
          f"{'█' * int(importances[idx] * 100)}")

# Compare with single tree
dt_importances = dt.feature_importances_
print("\nNote: Random Forest provides more stable feature importance than single trees!")

print("\n" + "=" * 60)
print("MODEL INSIGHTS")
print("=" * 60)

# Out-of-bag score (unique to Random Forest)
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)
print(f"Out-of-Bag Score: {rf_oob.oob_score_:.2%}")
print("(This is like built-in validation without needing separate validation set!)")

# Tree depth statistics
tree_depths = [tree.get_depth() for tree in rf_final.estimators_]
print(f"\nTree depth statistics:")
print(f"  Min depth: {min(tree_depths)}")
print(f"  Max depth: {max(tree_depths)}")
print(f"  Mean depth: {np.mean(tree_depths):.1f}")

# Visualizations
fig = plt.figure(figsize=(15, 10))

# Plot 1: n_estimators vs accuracy
ax1 = plt.subplot(2, 3, 1)
n_est_vals = [r['n_estimators'] for r in rf_results]
acc_vals = [r['accuracy'] for r in rf_results]
ax1.plot(n_est_vals, acc_vals, marker='o', linewidth=2, markersize=8)
ax1.axhline(y=dt_acc, color='r', linestyle='--', label='Single Tree')
ax1.set_xlabel('Number of Trees', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_title('Model Performance vs Number of Trees', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Feature importance
ax2 = plt.subplot(2, 3, 2)
y_pos = np.arange(len(feature_names))
ax2.barh(y_pos, importances[indices], alpha=0.8, color='skyblue', edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([feature_names[i] for i in indices])
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Feature Importance', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
ax3.set_xlabel('Predicted', fontsize=11)
ax3.set_ylabel('Actual', fontsize=11)
ax3.set_title('Confusion Matrix', fontweight='bold')

# Plot 4: Cross-validation scores
ax4 = plt.subplot(2, 3, 4)
ax4.bar(range(1, 6), cv_scores, alpha=0.8, color='lightgreen', edgecolor='black')
ax4.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean')
ax4.set_xlabel('Fold', fontsize=11)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_title('Cross-Validation Scores', fontweight='bold')
ax4.set_xticks(range(1, 6))
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Tree depth distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(tree_depths, bins=20, alpha=0.7, color='coral', edgecolor='black')
ax5.set_xlabel('Tree Depth', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Distribution of Tree Depths', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Prediction probability distribution
ax6 = plt.subplot(2, 3, 6)
proba_high_risk = y_pred_proba[:, 1]
ax6.hist(proba_high_risk[y_test == 0], bins=20, alpha=0.6,
         label='Actual Low Risk', color='blue', edgecolor='black')
ax6.hist(proba_high_risk[y_test == 1], bins=20, alpha=0.6,
         label='Actual High Risk', color='red', edgecolor='black')
ax6.set_xlabel('Predicted Probability (High Risk)', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Prediction Confidence Distribution', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/machine-learning/03-intermediate/random_forest_results.png', dpi=150)
print("\nVisualization saved to: 03-intermediate/random_forest_results.png")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. HOW RANDOM FOREST WORKS:
   a) Bootstrap Sampling: Random sample with replacement from training data
   b) Random Feature Selection: Each split considers random subset of features
   c) Build many trees independently
   d) Vote: Classification → majority vote, Regression → average

2. KEY HYPERPARAMETERS:
   - n_estimators: Number of trees (more is usually better, but slower)
   - max_depth: Maximum tree depth (prevents overfitting)
   - min_samples_split: Min samples to split node
   - max_features: Features to consider for each split
     * 'sqrt': √n_features (default for classification)
     * 'log2': log₂(n_features)
     * int/float: specific number or fraction

3. ADVANTAGES OVER SINGLE TREES:
   ✓ More accurate and robust
   ✓ Reduces overfitting through averaging
   ✓ Handles missing values well
   ✓ Out-of-bag error estimate (built-in validation)
   ✓ Less sensitive to outliers

4. PROS:
   ✓ Excellent performance out-of-the-box
   ✓ Works well with default parameters
   ✓ Provides feature importance
   ✓ Handles both classification and regression
   ✓ Parallelizable (fast training)

5. CONS:
   ✗ Less interpretable than single tree
   ✗ Larger model size (stores many trees)
   ✗ Slower prediction than single tree
   ✗ Can overfit on noisy datasets

6. OUT-OF-BAG (OOB) ERROR:
   - Each tree uses ~63% of data (due to bootstrap sampling)
   - Remaining ~37% used for validation
   - OOB score ≈ cross-validation accuracy
   - No need for separate validation set!

7. WHEN TO USE:
   ✓ Default choice for many problems
   ✓ When you need feature importance
   ✓ When you have complex non-linear relationships
   ✓ When interpretability is not critical

EXERCISE:
- Compare different max_features settings
- Experiment with tree depth limits
- Try ExtraTreesClassifier (even more randomness)
- Compare with Gradient Boosting
- Analyze feature interactions
""")
