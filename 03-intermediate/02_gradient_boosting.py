"""
Gradient Boosting - Sequential Ensemble Learning
=================================================
Unlike Random Forest (parallel), Gradient Boosting builds trees sequentially.
Each new tree corrects errors made by previous trees.

Often achieves state-of-the-art results in competitions!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.datasets import make_classification
from pathlib import Path

print("=" * 60)
print("GRADIENT BOOSTING - CUSTOMER CHURN PREDICTION")
print("=" * 60)

# Generate synthetic customer data
np.random.seed(42)
X, y = make_classification(
    n_samples=2000,
    n_features=15,
    n_informative=12,
    n_redundant=3,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced dataset
    flip_y=0.05,
    random_state=42
)

feature_names = [
    'tenure', 'monthly_charges', 'total_charges', 'age',
    'num_products', 'support_calls', 'contract_length',
    'payment_delay', 'usage_minutes', 'data_usage',
    'account_age', 'complaints', 'satisfaction_score',
    'referrals', 'promotions_used'
]

print(f"Dataset: {len(X)} customers")
print(f"Features: {len(feature_names)}")
print(f"Target: 0 = Retained, 1 = Churned")
print(f"Class distribution: {np.bincount(y)} (imbalanced!)\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Train different models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Get prediction probabilities for ROC curve
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None

    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_proba
    }

    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print(f"  Difference: {(train_acc - test_acc):.2%} "
          f"{'(overfitting!)' if train_acc - test_acc > 0.05 else '(good!)'}")

# Detailed analysis of Gradient Boosting
print("\n" + "=" * 60)
print("GRADIENT BOOSTING - DETAILED ANALYSIS")
print("=" * 60)

gb = results['Gradient Boosting']['model']
y_pred = results['Gradient Boosting']['y_pred']

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Retained', 'Churned']))

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importances = gb.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 most important features:")
for i in range(10):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f} "
          f"{'█' * int(importances[idx] * 200)}")

print("\n" + "=" * 60)
print("TUNING NUMBER OF BOOSTING ITERATIONS")
print("=" * 60)

# Train with different numbers of estimators
n_estimators_range = range(10, 201, 10)
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    gb_temp = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_temp.fit(X_train, y_train)

    train_scores.append(accuracy_score(y_train, gb_temp.predict(X_train)))
    test_scores.append(accuracy_score(y_test, gb_temp.predict(X_test)))

best_n_est = list(n_estimators_range)[np.argmax(test_scores)]
print(f"Best n_estimators: {best_n_est}")
print(f"Best test accuracy: {max(test_scores):.2%}")

print("\n" + "=" * 60)
print("LEARNING RATE IMPACT")
print("=" * 60)

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
lr_results = []

for lr in learning_rates:
    gb_lr = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb_lr.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, gb_lr.predict(X_test))
    lr_results.append(test_acc)

    print(f"Learning rate {lr:.2f} → Test Accuracy: {test_acc:.2%}")

# Visualizations
fig = plt.figure(figsize=(16, 10))

# Plot 1: Model comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
test_accs = [results[name]['test_acc'] for name in model_names]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(model_names, test_accs, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.set_title('Model Comparison', fontweight='bold')
ax1.set_ylim([min(test_accs) - 0.05, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Feature importance
ax2 = plt.subplot(2, 3, 2)
top_10_indices = indices[:10]
top_10_importances = importances[top_10_indices]
top_10_features = [feature_names[i] for i in top_10_indices]
y_pos = np.arange(len(top_10_features))
ax2.barh(y_pos, top_10_importances, alpha=0.8, color='coral', edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_10_features, fontsize=9)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Top 10 Feature Importance', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: ROC Curves
ax3 = plt.subplot(2, 3, 3)
for name, result in results.items():
    if result['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax3.set_xlabel('False Positive Rate', fontsize=11)
ax3.set_ylabel('True Positive Rate', fontsize=11)
ax3.set_title('ROC Curves', fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# Plot 4: n_estimators vs accuracy
ax4 = plt.subplot(2, 3, 4)
ax4.plot(n_estimators_range, train_scores, label='Training', linewidth=2, marker='o', markersize=4)
ax4.plot(n_estimators_range, test_scores, label='Testing', linewidth=2, marker='s', markersize=4)
ax4.axvline(x=best_n_est, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_n_est}')
ax4.set_xlabel('Number of Trees', fontsize=11)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_title('Boosting Iterations vs Performance', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Learning rate impact
ax5 = plt.subplot(2, 3, 5)
ax5.plot(learning_rates, lr_results, marker='o', linewidth=2, markersize=10, color='purple')
ax5.set_xlabel('Learning Rate', fontsize=11)
ax5.set_ylabel('Test Accuracy', fontsize=11)
ax5.set_title('Learning Rate Impact', fontweight='bold')
ax5.set_xscale('log')
ax5.grid(True, alpha=0.3)

# Plot 6: Training vs Test comparison
ax6 = plt.subplot(2, 3, 6)
model_names = list(results.keys())
train_accs = [results[name]['train_acc'] for name in model_names]
test_accs = [results[name]['test_acc'] for name in model_names]
x = np.arange(len(model_names))
width = 0.35

ax6.bar(x - width/2, train_accs, width, label='Training', alpha=0.8, color='lightblue', edgecolor='black')
ax6.bar(x + width/2, test_accs, width, label='Testing', alpha=0.8, color='lightcoral', edgecolor='black')
ax6.set_ylabel('Accuracy', fontsize=11)
ax6.set_title('Train vs Test Accuracy', fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(model_names, fontsize=9)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = Path(__file__).parent / 'gradient_boosting_results.png'
plt.savefig(output_path, dpi=150)
print(f"
Visualization saved to: {output_path}")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. HOW GRADIENT BOOSTING WORKS:
   Step 1: Train initial model (often just predicting mean)
   Step 2: Calculate residuals (errors) from predictions
   Step 3: Train new model to predict these residuals
   Step 4: Add this model to ensemble (with learning rate scaling)
   Step 5: Repeat steps 2-4 for n_estimators iterations

   Final prediction = sum of all tree predictions × learning_rate

2. KEY HYPERPARAMETERS:
   - n_estimators: Number of boosting stages (trees to build)
     * More = better fit but slower, risk of overfitting
     * Typical: 100-500

   - learning_rate (shrinkage): Scales contribution of each tree
     * Lower = more robust but needs more trees
     * Typical: 0.01-0.1
     * Trade-off: n_estimators ↑ when learning_rate ↓

   - max_depth: Depth of individual trees
     * Typical: 3-5 (shallow trees work well!)
     * Boosting works better with weak learners

   - subsample: Fraction of samples for each tree (stochastic GB)
     * < 1.0 adds randomness, reduces overfitting
     * Typical: 0.8

3. GRADIENT BOOSTING VS RANDOM FOREST:

   Random Forest:
   ✓ Trains trees in parallel (faster)
   ✓ Less prone to overfitting
   ✓ More robust to hyperparameters

   Gradient Boosting:
   ✓ Often higher accuracy
   ✓ Better on smaller datasets
   ✗ Slower training (sequential)
   ✗ More sensitive to hyperparameters
   ✗ Can overfit if not tuned properly

4. PROS:
   ✓ State-of-the-art performance on many tasks
   ✓ Handles mixed data types
   ✓ Robust to outliers (with proper loss function)
   ✓ Feature importance available
   ✓ No feature scaling needed

5. CONS:
   ✗ Sensitive to hyperparameters
   ✗ Slow training (sequential)
   ✗ Can overfit easily
   ✗ Less interpretable than single trees

6. VARIANTS:
   - XGBoost: Extremely popular, optimized implementation
   - LightGBM: Faster, uses histogram-based learning
   - CatBoost: Handles categorical features natively

   These are often used in ML competitions!

7. BEST PRACTICES:
   - Start with learning_rate=0.1, n_estimators=100
   - Use early stopping with validation set
   - Tune max_depth (try 3-7)
   - Monitor train vs validation scores
   - Use subsample < 1.0 for large datasets

EXERCISE:
- Implement early stopping
- Try XGBoost or LightGBM
- Experiment with different loss functions
- Compare with AdaBoost
- Tune hyperparameters with GridSearchCV
""")
