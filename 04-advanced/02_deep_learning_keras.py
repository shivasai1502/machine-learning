"""
Deep Learning with Keras/TensorFlow
====================================
Keras is the industry-standard deep learning library.
It's simple to use but powerful enough for research.

NOTE: Install TensorFlow first: pip install tensorflow
"""

print("=" * 60)
print("DEEP LEARNING WITH KERAS")
print("=" * 60)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_digits
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}\n")

except ImportError:
    print("\n" + "!" * 60)
    print("TensorFlow not installed!")
    print("Install with: pip install tensorflow")
    print("!" * 60)
    print("\nThis script demonstrates what you'll learn:")
    print("""
    1. Building neural networks with Keras Sequential API
    2. Using different layer types (Dense, Dropout, BatchNormalization)
    3. Compiling models (optimizer, loss, metrics)
    4. Training with callbacks (EarlyStopping, LearningRateScheduler)
    5. Evaluating and visualizing results
    6. Saving and loading models

    KEY KERAS CONCEPTS:

    MODEL CREATION:
    ```python
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    ```

    COMPILATION:
    ```python
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    ```

    TRAINING:
    ```python
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    ```

    POPULAR OPTIMIZERS:
    - Adam: Default choice, works well
    - SGD: With momentum, good for large datasets
    - RMSprop: Good for RNNs
    - AdamW: Adam with weight decay

    LOSS FUNCTIONS:
    - Binary classification: 'binary_crossentropy'
    - Multi-class (one-hot): 'categorical_crossentropy'
    - Multi-class (integers): 'sparse_categorical_crossentropy'
    - Regression: 'mse' or 'mae'

    CALLBACKS:
    - EarlyStopping: Stop when validation stops improving
    - ModelCheckpoint: Save best model
    - ReduceLROnPlateau: Reduce learning rate when stuck
    - TensorBoard: Visualization

    REGULARIZATION:
    - Dropout: Randomly disable neurons
    - L1/L2: Weight penalties
    - BatchNormalization: Normalize layer inputs

    INSTALL AND RUN THIS SCRIPT TO SEE FULL EXAMPLES!
    """)
    exit()

print("=" * 60)
print("EXAMPLE 1: HANDWRITTEN DIGIT CLASSIFICATION")
print("=" * 60)

# Load digits dataset (8x8 images)
digits = load_digits()
X = digits.data  # 64 features (8x8 pixels flattened)
y = digits.target  # 10 classes (digits 0-9)

print(f"Dataset: {X.shape[0]} images")
print(f"Image size: 8x8 pixels")
print(f"Classes: 10 (digits 0-9)")
print(f"Features: {X.shape[1]}\n")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

print("=" * 60)
print("BUILDING THE MODEL")
print("=" * 60)

# Create a deep neural network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,), name='hidden1'),
    layers.Dropout(0.3, name='dropout1'),
    layers.Dense(64, activation='relu', name='hidden2'),
    layers.Dropout(0.2, name='dropout2'),
    layers.Dense(32, activation='relu', name='hidden3'),
    layers.Dense(10, activation='softmax', name='output')
])

print("Model architecture:")
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled!")
print("  Optimizer: Adam (lr=0.001)")
print("  Loss: Sparse Categorical Crossentropy")
print("  Metrics: Accuracy\n")

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("=" * 60)
print("TRAINING THE MODEL")
print("=" * 60)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}\n")

# Predictions
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Visualizations
fig = plt.figure(figsize=(16, 10))

# Plot 1: Training history - Loss
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Model Loss Over Time', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training history - Accuracy
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Model Accuracy Over Time', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_xlabel('Predicted', fontsize=11)
ax3.set_ylabel('Actual', fontsize=11)
ax3.set_title('Confusion Matrix', fontweight='bold')

# Plot 4-9: Sample predictions
for i in range(6):
    ax = plt.subplot(2, 3, i + 4)
    idx = np.random.randint(0, len(X_test))

    image = X_test[idx].reshape(8, 8)
    true_label = y_test[idx]
    pred_label = y_pred_classes[idx]
    confidence = y_pred[idx][pred_label]

    ax.imshow(image, cmap='gray')
    ax.axis('off')

    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}, Pred: {pred_label}\n'
                f'Confidence: {confidence:.2%}',
                fontsize=9, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('/home/user/machine-learning/04-advanced/keras_results.png', dpi=150)
print("\nVisualization saved to: 04-advanced/keras_results.png")

# Save the model
model.save('/home/user/machine-learning/04-advanced/digit_classifier.keras')
print("\nModel saved to: 04-advanced/digit_classifier.keras")

print("\n" + "=" * 60)
print("LOADING AND USING SAVED MODEL")
print("=" * 60)

# Load the model
loaded_model = keras.models.load_model('/home/user/machine-learning/04-advanced/digit_classifier.keras')

# Make predictions with loaded model
sample_predictions = loaded_model.predict(X_test_scaled[:5], verbose=0)
print("Predictions from loaded model:")
for i, pred in enumerate(sample_predictions):
    print(f"Sample {i+1}: True={y_test[i]}, "
          f"Predicted={np.argmax(pred)}, "
          f"Confidence={np.max(pred):.2%}")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. KERAS WORKFLOW:
   a) Prepare data (scale, split)
   b) Build model (Sequential or Functional API)
   c) Compile (optimizer, loss, metrics)
   d) Train with fit()
   e) Evaluate and predict
   f) Save/load model

2. MODEL BUILDING:

   Sequential API (simple, linear):
   ```python
   model = keras.Sequential([
       layers.Dense(128, activation='relu'),
       layers.Dense(10, activation='softmax')
   ])
   ```

   Functional API (complex, multi-input/output):
   ```python
   inputs = keras.Input(shape=(784,))
   x = layers.Dense(128, activation='relu')(inputs)
   outputs = layers.Dense(10, activation='softmax')(x)
   model = keras.Model(inputs=inputs, outputs=outputs)
   ```

3. COMMON LAYERS:

   Dense: Fully connected layer
   - Most common, uses: hidden layers, output layer

   Dropout: Regularization
   - Randomly sets inputs to 0
   - Prevents overfitting
   - Typical rates: 0.2-0.5

   BatchNormalization: Normalize layer inputs
   - Speeds up training
   - Reduces internal covariate shift
   - Place after Dense, before activation

   Conv2D: Convolutional layer (for images)
   LSTM/GRU: Recurrent layers (for sequences)

4. ACTIVATION FUNCTIONS:

   Hidden layers:
   - ReLU: Default choice
   - LeakyReLU: Prevents dying ReLU
   - ELU: Smooth, allows negative values

   Output layer:
   - Sigmoid: Binary classification
   - Softmax: Multi-class classification
   - Linear: Regression

5. OPTIMIZERS:

   Adam (Adaptive Moment Estimation):
   ✓ Default choice
   ✓ Combines momentum + RMSprop
   ✓ Works well without tuning

   SGD (Stochastic Gradient Descent):
   ✓ With momentum: good for large datasets
   ✓ More stable, generalizes better
   ✗ Requires learning rate tuning

   Learning rates:
   - Too high: Training unstable, diverges
   - Too low: Training too slow, gets stuck
   - Typical: 0.001 (1e-3) for Adam

6. CALLBACKS:

   EarlyStopping:
   - Stops when validation metric stops improving
   - restore_best_weights=True recommended

   ModelCheckpoint:
   - Saves model during training
   - Keep best model based on metric

   ReduceLROnPlateau:
   - Reduces learning rate when stuck
   - Helps fine-tune training

   TensorBoard:
   - Real-time training visualization
   - Great for experiments

7. REGULARIZATION:

   Dropout:
   ```python
   layers.Dropout(0.3)  # Drop 30% of inputs
   ```

   L1/L2 (Weight Decay):
   ```python
   layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))
   ```

   BatchNormalization:
   ```python
   layers.BatchNormalization()
   ```

8. BEST PRACTICES:

   ✓ Always normalize/scale inputs
   ✓ Use validation set (validation_split=0.2)
   ✓ Monitor both train and val metrics
   ✓ Use callbacks (EarlyStopping, ReduceLR)
   ✓ Start simple, add complexity
   ✓ Save your best models
   ✓ Use proper activation functions
   ✓ Batch size: 32-128 typically

   ✗ Don't train too long without validation
   ✗ Don't use test set for hyperparameter tuning
   ✗ Don't forget to scale data

9. NEXT STEPS:

   - CNNs (Convolutional Neural Networks) for images
   - RNNs/LSTMs for sequential data
   - Transfer learning with pre-trained models
   - Advanced architectures (ResNet, U-Net, etc.)
   - Autoencoders for unsupervised learning
   - GANs (Generative Adversarial Networks)

EXERCISE:
- Try different architectures (wider, deeper)
- Experiment with dropout rates
- Add BatchNormalization
- Try different optimizers and learning rates
- Implement custom callbacks
- Build a CNN for image classification
""")

