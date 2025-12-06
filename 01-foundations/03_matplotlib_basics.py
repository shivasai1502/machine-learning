"""
Matplotlib Basics - Data Visualization for ML
==============================================
Visualization is crucial for understanding data and model performance.
"""

import matplotlib.pyplot as plt
import numpy as np

print("Generating various plots for Machine Learning visualization...")
print("=" * 60)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. Line Plot
ax1 = plt.subplot(2, 3, 1)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
ax1.plot(x, y1, label='sin(x)', linewidth=2)
ax1.plot(x, y2, label='cos(x)', linewidth=2)
ax1.set_title('Line Plot - Function Visualization')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# 2. Scatter Plot (common in ML for showing relationships)
ax2 = plt.subplot(2, 3, 2)
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
ax2.scatter(x_scatter, y_scatter, alpha=0.6, c='blue', edgecolors='black')
ax2.set_title('Scatter Plot - Feature Relationship')
ax2.set_xlabel('Feature X')
ax2.set_ylabel('Feature Y')
ax2.grid(True)

# 3. Histogram (distribution analysis)
ax3 = plt.subplot(2, 3, 3)
data = np.random.randn(1000)
ax3.hist(data, bins=30, edgecolor='black', alpha=0.7)
ax3.set_title('Histogram - Data Distribution')
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

# 4. Bar Plot (comparing categories)
ax4 = plt.subplot(2, 3, 4)
categories = ['Model A', 'Model B', 'Model C', 'Model D']
accuracy = [0.85, 0.92, 0.88, 0.95]
bars = ax4.bar(categories, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
               edgecolor='black')
ax4.set_title('Bar Plot - Model Comparison')
ax4.set_ylabel('Accuracy')
ax4.set_ylim([0.8, 1.0])
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# 5. Box Plot (showing statistical distribution)
ax5 = plt.subplot(2, 3, 5)
data_box = [np.random.normal(0, std, 100) for std in range(1, 4)]
ax5.boxplot(data_box, labels=['Feature 1', 'Feature 2', 'Feature 3'])
ax5.set_title('Box Plot - Feature Distributions')
ax5.set_ylabel('Value')
ax5.grid(True, alpha=0.3)

# 6. Heatmap (correlation matrix - very important in ML!)
ax6 = plt.subplot(2, 3, 6)
correlation_matrix = np.random.rand(5, 5)
# Make it symmetric
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1)

im = ax6.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax6.set_title('Heatmap - Feature Correlation')
ax6.set_xticks(range(5))
ax6.set_yticks(range(5))
ax6.set_xticklabels([f'F{i}' for i in range(1, 6)])
ax6.set_yticklabels([f'F{i}' for i in range(1, 6)])

# Add colorbar
plt.colorbar(im, ax=ax6)

# Add correlation values
for i in range(5):
    for j in range(5):
        text = ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('/home/user/machine-learning/01-foundations/visualization_examples.png', dpi=150, bbox_inches='tight')
print("Saved visualization examples to: 01-foundations/visualization_examples.png")

# Create a second figure for learning curves (important for ML!)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Simulated learning curves
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-epochs/10) + 0.1
val_loss = 2.5 * np.exp(-epochs/10) + 0.3 + 0.1 * np.random.rand(50)

ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True)

# Simulated accuracy curves
train_acc = 1 - 0.9 * np.exp(-epochs/10)
val_acc = 1 - 0.9 * np.exp(-epochs/10) - 0.05

ax2.plot(epochs, train_acc, label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training vs Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('/home/user/machine-learning/01-foundations/learning_curves.png', dpi=150, bbox_inches='tight')
print("Saved learning curves to: 01-foundations/learning_curves.png")

print("\n" + "=" * 60)
print("COMMON PLOT TYPES IN MACHINE LEARNING:")
print("=" * 60)
print("""
1. Line Plot: Training/validation curves, convergence
2. Scatter Plot: Feature relationships, clustering results
3. Histogram: Data distribution, residual analysis
4. Bar Plot: Model comparison, feature importance
5. Box Plot: Outlier detection, feature statistics
6. Heatmap: Correlation matrix, confusion matrix
7. Learning Curves: Model performance over time
""")

print("\nPlots saved! Check the generated PNG files.")
print("\nTo display plots interactively, add 'plt.show()' at the end.")
