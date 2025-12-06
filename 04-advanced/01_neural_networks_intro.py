"""
Neural Networks Introduction
=============================
Neural networks are the foundation of deep learning.
They're inspired by the human brain and can learn complex patterns.

Start here before diving into CNNs, RNNs, or Transformers!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_moons, make_circles

print("=" * 60)
print("NEURAL NETWORKS - FROM SCRATCH TO SKLEARN")
print("=" * 60)

print("\n" + "=" * 60)
print("PART 1: UNDERSTANDING THE BUILDING BLOCKS")
print("=" * 60)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Visualize activation functions
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x), linewidth=2, color='blue')
plt.title('Sigmoid: σ(x) = 1 / (1 + e^(-x))', fontweight='bold')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.subplot(1, 3, 2)
plt.plot(x, relu(x), linewidth=2, color='red')
plt.title('ReLU: max(0, x)', fontweight='bold')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.subplot(1, 3, 3)
plt.plot(x, tanh(x), linewidth=2, color='green')
plt.title('Tanh: (e^x - e^(-x)) / (e^x + e^(-x))', fontweight='bold')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
output_path = Path(__file__).parent / 'activation_functions.png'
plt.savefig(output_path, dpi=150)
print(f"\nActivation functions visualization saved to: {output_path}")

print("\n" + "=" * 60)
print("PART 2: SIMPLE NEURAL NETWORK (2 LAYERS)")
print("=" * 60)

# Generate non-linear dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

print(f"Dataset: {len(X)} samples")
print("Classes: 2 (non-linearly separable)")

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Simple neural network: Input → Hidden(5 neurons) → Output
nn_simple = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

print("\nNetwork architecture:")
print("  Input layer: 2 neurons (2 features)")
print("  Hidden layer: 5 neurons (ReLU activation)")
print("  Output layer: 1 neuron (sigmoid)")

nn_simple.fit(X_train_scaled, y_train)

# Calculate parameters after fitting
n_params = sum(p.size for p in nn_simple.coefs_) + sum(p.size for p in nn_simple.intercepts_)
print(f"  Total parameters: {n_params}")

y_pred = nn_simple.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.2%}")
print(f"Training iterations: {nn_simple.n_iter_}")

print("\n" + "=" * 60)
print("PART 3: COMPARING NETWORK ARCHITECTURES")
print("=" * 60)

# Test different architectures
architectures = [
    (5,),           # 1 hidden layer, 5 neurons
    (10,),          # 1 hidden layer, 10 neurons
    (20,),          # 1 hidden layer, 20 neurons
    (10, 10),       # 2 hidden layers, 10 neurons each
    (20, 10),       # 2 hidden layers
    (50, 25, 10),   # 3 hidden layers
]

results = []

for arch in architectures:
    nn = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    nn.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, nn.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, nn.predict(X_test_scaled))

    n_params = sum(p.size for p in nn.coefs_) + sum(p.size for p in nn.intercepts_)

    results.append({
        'architecture': arch,
        'arch_str': '-'.join(map(str, arch)),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_params': n_params,
        'n_iter': nn.n_iter_
    })

    print(f"Architecture {str(arch):15s} → Train: {train_acc:.2%}, "
          f"Test: {test_acc:.2%}, Params: {n_params:4d}")

best = max(results, key=lambda x: x['test_acc'])
print(f"\nBest architecture: {best['architecture']} (Test: {best['test_acc']:.2%})")

print("\n" + "=" * 60)
print("PART 4: COMPARING ACTIVATION FUNCTIONS")
print("=" * 60)

activations = ['relu', 'tanh', 'logistic']
activation_results = []

for activation in activations:
    nn = MLPClassifier(
        hidden_layer_sizes=(20, 10),
        activation=activation,
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    nn.fit(X_train_scaled, y_train)

    test_acc = accuracy_score(y_test, nn.predict(X_test_scaled))

    activation_results.append({
        'activation': activation,
        'accuracy': test_acc
    })

    print(f"{activation:10s} → Test Accuracy: {test_acc:.2%}")

print("\n" + "=" * 60)
print("PART 5: SOLVING COMPLEX PATTERNS")
print("=" * 60)

# Create more complex dataset (circles)
X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

print("Testing on concentric circles (harder problem)...")

# Deep network
nn_deep = MLPClassifier(
    hidden_layer_sizes=(50, 25, 10),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

nn_deep.fit(X_train_c_scaled, y_train_c)
circles_acc = accuracy_score(y_test_c, nn_deep.predict(X_test_c_scaled))

print(f"Deep NN on circles: {circles_acc:.2%}")

# Helper function to avoid repeated code
h_step = 0.02

# Visualizations
fig = plt.figure(figsize=(16, 12))

# Plot 1: Architecture comparison
ax1 = plt.subplot(3, 3, 1)
arch_names = [r['arch_str'] for r in results]
test_accs = [r['test_acc'] for r in results]
ax1.bar(range(len(arch_names)), test_accs, alpha=0.8, color='skyblue', edgecolor='black')
ax1.set_xticks(range(len(arch_names)))
ax1.set_xticklabels(arch_names, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.set_title('Architecture Comparison', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Parameters vs Accuracy
ax2 = plt.subplot(3, 3, 2)
params = [r['n_params'] for r in results]
ax2.scatter(params, test_accs, s=100, alpha=0.7, c=test_accs, cmap='viridis', edgecolors='black')
ax2.set_xlabel('Number of Parameters', fontsize=11)
ax2.set_ylabel('Test Accuracy', fontsize=11)
ax2.set_title('Model Complexity vs Performance', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Activation function comparison
ax3 = plt.subplot(3, 3, 3)
act_names = [r['activation'] for r in activation_results]
act_accs = [r['accuracy'] for r in activation_results]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax3.bar(act_names, act_accs, color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('Test Accuracy', fontsize=11)
ax3.set_title('Activation Function Comparison', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontsize=10)

# Decision boundaries for different architectures
datasets = [
    (X_train, y_train, X_test, y_test, "Moons Dataset"),
    (X_train_c, y_train_c, X_test_c, y_test_c, "Circles Dataset")
]

plot_idx = 4
for X_tr, y_tr, X_te, y_te, title in datasets:
    # Scale data
    scaler_temp = StandardScaler()
    X_tr_scaled = scaler_temp.fit_transform(X_tr)

    for hidden_size in [(5,), (20, 10), (50, 25, 10)]:
        ax = plt.subplot(3, 3, plot_idx)

        nn_temp = MLPClassifier(
            hidden_layer_sizes=hidden_size,
            activation='relu',
            max_iter=2000,
            random_state=42
        )
        nn_temp.fit(X_tr_scaled, y_tr)

        # Create decision boundary
        x_min, x_max = X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5
        y_min, y_max = X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h_step),
                             np.arange(y_min, y_max, h_step))

        Z = nn_temp.predict(scaler_temp.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap='coolwarm',
                  edgecolors='black', s=30, alpha=0.8)

        test_acc = accuracy_score(y_te, nn_temp.predict(scaler_temp.transform(X_te)))
        arch_str = '-'.join(map(str, hidden_size))
        ax.set_title(f'{title}: {arch_str}\nAcc: {test_acc:.2%}',
                    fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plot_idx += 1

plt.tight_layout()
output_path = Path(__file__).parent / 'neural_networks_results.png'
plt.savefig(output_path, dpi=150)
print(f"\nVisualization saved to: {output_path}")

print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
1. NEURAL NETWORK COMPONENTS:

   Neuron: output = activation(Σ(weight × input) + bias)

   Layers:
   - Input layer: Receives features
   - Hidden layers: Learn representations
   - Output layer: Makes predictions

2. ACTIVATION FUNCTIONS:

   ReLU (Rectified Linear Unit): f(x) = max(0, x)
   ✓ Most popular for hidden layers
   ✓ Solves vanishing gradient problem
   ✓ Fast to compute
   ✗ Can "die" (always outputs 0)

   Sigmoid: f(x) = 1 / (1 + e^(-x))
   ✓ Output in [0, 1], good for probabilities
   ✗ Vanishing gradients
   ✗ Not zero-centered

   Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   ✓ Output in [-1, 1], zero-centered
   ✗ Still has vanishing gradient

   When to use:
   - Hidden layers: ReLU (default), Leaky ReLU, ELU
   - Output layer: Sigmoid (binary), Softmax (multi-class), Linear (regression)

3. FORWARD PROPAGATION:
   Data flows forward through network:
   Input → Hidden1 → Hidden2 → ... → Output

4. BACKPROPAGATION:
   Calculate gradients backward:
   Loss → ∂Output → ∂Hidden2 → ∂Hidden1 → Update weights

5. HYPERPARAMETERS:

   Architecture:
   - hidden_layer_sizes: Network depth and width
   - activation: Activation function

   Training:
   - learning_rate: Step size for weight updates (default: 0.001)
   - batch_size: Samples per gradient update
   - max_iter/epochs: Training iterations

   Regularization:
   - alpha: L2 penalty (weight decay)
   - dropout: Randomly disable neurons

6. OPTIMIZATION ALGORITHMS:

   SGD: Stochastic Gradient Descent (basic)
   Adam: Adaptive Moment Estimation (most popular)
   RMSprop: Root Mean Square Propagation
   AdaGrad: Adaptive Gradient

7. COMMON ISSUES:

   Overfitting:
   - Network too large
   - Training too long
   Solutions: Regularization, dropout, early stopping

   Underfitting:
   - Network too small
   - Not enough training
   Solutions: More layers/neurons, train longer

   Vanishing Gradients:
   - Gradients become too small
   - Deep networks with sigmoid/tanh
   Solutions: ReLU, batch normalization, skip connections

   Exploding Gradients:
   - Gradients become too large
   Solutions: Gradient clipping, proper initialization

8. BEST PRACTICES:

   ✓ Always scale/normalize input features
   ✓ Start simple, add complexity gradually
   ✓ Use ReLU for hidden layers
   ✓ Monitor training & validation loss
   ✓ Use early stopping
   ✓ Try different learning rates (0.001, 0.01, 0.0001)
   ✓ Use batch normalization for deep networks

9. WHEN TO USE NEURAL NETWORKS:

   ✓ Large datasets (100k+ samples)
   ✓ Complex non-linear patterns
   ✓ Images, text, sequences
   ✓ Feature learning is important

   ✗ Small datasets (try tree-based models)
   ✗ Need interpretability
   ✗ Limited computational resources

NEXT STEPS:
- Learn TensorFlow/Keras or PyTorch
- Study Convolutional Neural Networks (CNNs) for images
- Study Recurrent Neural Networks (RNNs) for sequences
- Explore modern architectures (ResNet, Transformers)

EXERCISE:
- Experiment with different network depths
- Try different optimizers and learning rates
- Implement early stopping
- Add dropout for regularization
- Visualize weights and activations
""")
