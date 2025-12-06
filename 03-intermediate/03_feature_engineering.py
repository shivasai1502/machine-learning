"""
Feature Engineering - Creating Better Features
===============================================
"Feature engineering is the art of creating informative features from raw data"

Often, feature engineering makes the biggest difference in model performance!
Good features > Complex models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE

print("=" * 60)
print("FEATURE ENGINEERING - COMPLETE GUIDE")
print("=" * 60)

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'years_employed': np.random.randint(0, 40, n_samples),
    'num_accounts': np.random.randint(1, 10, n_samples),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'has_car': np.random.choice([0, 1], n_samples),
    'has_property': np.random.choice([0, 1], n_samples),
})

# Create target based on features (with some noise)
data['approved'] = (
    (data['credit_score'] > 650) &
    (data['income'] > 50000) &
    (data['years_employed'] > 2)
).astype(int)

# Add noise
noise = np.random.random(n_samples) < 0.1
data.loc[noise, 'approved'] = 1 - data.loc[noise, 'approved']

print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nTarget distribution:")
print(data['approved'].value_counts())

print("\n" + "=" * 60)
print("1. HANDLING CATEGORICAL VARIABLES")
print("=" * 60)

print("\nOriginal categorical columns: city, education")

# Label Encoding (for ordinal data)
education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
data['education_encoded'] = data['education'].map(education_mapping)

print("\nLabel Encoding (education):")
print(data[['education', 'education_encoded']].drop_duplicates().sort_values('education_encoded'))

# One-Hot Encoding (for nominal data)
city_dummies = pd.get_dummies(data['city'], prefix='city')
data = pd.concat([data, city_dummies], axis=1)

print("\nOne-Hot Encoding (city):")
print(f"New columns created: {city_dummies.columns.tolist()}")
print(city_dummies.head())

print("\n" + "=" * 60)
print("2. NUMERICAL TRANSFORMATIONS")
print("=" * 60)

# Log transformation (for skewed features)
data['income_log'] = np.log1p(data['income'])  # log1p = log(1+x) to handle 0s

# Square root transformation
data['age_sqrt'] = np.sqrt(data['age'])

# Binning continuous variables
data['age_group'] = pd.cut(data['age'],
                           bins=[0, 25, 40, 60, 100],
                           labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

# Convert to numeric for modeling
age_group_encoded = pd.get_dummies(data['age_group'], prefix='age_group')
data = pd.concat([data, age_group_encoded], axis=1)

print("Transformations created:")
print(f"  - income_log: log transform for skewed data")
print(f"  - age_sqrt: square root transform")
print(f"  - age_group: binned age into categories")

print("\n" + "=" * 60)
print("3. FEATURE SCALING")
print("=" * 60)

# Select numerical features
numerical_features = ['age', 'income', 'credit_score', 'years_employed']

print("\nOriginal scale:")
print(data[numerical_features].describe())

# StandardScaler: mean=0, std=1
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data[numerical_features])
df_standardized = pd.DataFrame(data_standardized, columns=[f'{col}_standard' for col in numerical_features])

print("\nAfter StandardScaler (mean=0, std=1):")
print(df_standardized.describe())

# MinMaxScaler: range [0, 1]
scaler_minmax = MinMaxScaler()
data_normalized = scaler_minmax.fit_transform(data[numerical_features])
df_normalized = pd.DataFrame(data_normalized, columns=[f'{col}_minmax' for col in numerical_features])

print("\nAfter MinMaxScaler (range [0, 1]):")
print(df_normalized.describe())

print("\n" + "=" * 60)
print("4. CREATING INTERACTION FEATURES")
print("=" * 60)

# Manual interactions
data['income_per_age'] = data['income'] / (data['age'] + 1)
data['debt_to_income'] = data['num_accounts'] / (data['income'] / 1000)
data['credit_x_employment'] = data['credit_score'] * data['years_employed']

print("Interaction features created:")
print("  - income_per_age: income / age")
print("  - debt_to_income: accounts / (income/1000)")
print("  - credit_x_employment: credit_score × years_employed")

# Polynomial features
print("\nPolynomial Features:")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_features = poly.fit_transform(data[['credit_score', 'income']].values[:5])
poly_names = poly.get_feature_names_out(['credit_score', 'income'])

print(f"Original features: credit_score, income")
print(f"After polynomial (degree=2): {poly_names}")
print("\nExample values:")
print(poly_features[:3])

print("\n" + "=" * 60)
print("5. AGGREGATION FEATURES")
print("=" * 60)

# Group statistics
data['total_assets'] = data['has_car'] + data['has_property']
data['income_to_credit_ratio'] = data['income'] / (data['credit_score'] + 1)
data['employment_stability'] = np.where(data['years_employed'] > 5, 1, 0)

print("Aggregation features:")
print("  - total_assets: sum of car and property ownership")
print("  - income_to_credit_ratio: income / credit_score")
print("  - employment_stability: binary (employed > 5 years)")

print("\n" + "=" * 60)
print("6. FEATURE SELECTION")
print("=" * 60)

# Prepare data for modeling
feature_cols = [
    'age', 'income', 'credit_score', 'years_employed', 'num_accounts',
    'has_car', 'has_property', 'education_encoded',
    'income_per_age', 'debt_to_income', 'credit_x_employment',
    'total_assets', 'employment_stability'
] + city_dummies.columns.tolist()

X = data[feature_cols].fillna(0)
y = data['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Univariate Feature Selection
print("\nMethod 1: Univariate Feature Selection (SelectKBest)")
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
selected_features = X.columns[selector.get_support()].tolist()

print(f"Top 10 features by F-statistic:")
scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
print(scores.head(10).to_string(index=False))

# Method 2: Recursive Feature Elimination (RFE)
print("\nMethod 2: Recursive Feature Elimination (RFE)")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X_train, y_train)

print(f"Top 10 features by RFE:")
rfe_features = X.columns[rfe.support_].tolist()
for i, feat in enumerate(rfe_features, 1):
    print(f"  {i:2d}. {feat}")

# Method 3: Feature Importance from Random Forest
print("\nMethod 3: Feature Importance (Random Forest)")
rf.fit(X_train, y_train)
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 features by importance:")
print(importances.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("7. IMPACT ON MODEL PERFORMANCE")
print("=" * 60)

# Test different feature sets
experiments = {
    'Original Features': ['age', 'income', 'credit_score', 'years_employed', 'num_accounts'],
    'With Categorical': ['age', 'income', 'credit_score', 'years_employed', 'num_accounts',
                        'has_car', 'has_property', 'education_encoded'] + city_dummies.columns.tolist(),
    'With Interactions': feature_cols,
    'Selected Features (RFE)': rfe_features
}

results = []

for name, features in experiments.items():
    X_exp = data[features].fillna(0)
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_exp, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_exp, y_train_exp)

    train_acc = accuracy_score(y_train_exp, rf.predict(X_train_exp))
    test_acc = accuracy_score(y_test_exp, rf.predict(X_test_exp))

    results.append({
        'Feature Set': name,
        'Num Features': len(features),
        'Train Acc': train_acc,
        'Test Acc': test_acc
    })

    print(f"\n{name}:")
    print(f"  Features: {len(features)}")
    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy: {test_acc:.2%}")

# Visualizations
fig = plt.figure(figsize=(16, 10))

# Plot 1: Numerical distributions
ax1 = plt.subplot(2, 3, 1)
data[numerical_features].hist(bins=30, ax=ax1, alpha=0.7, edgecolor='black')
plt.suptitle('Original Feature Distributions', y=1.02, fontweight='bold')

# Plot 2: Feature correlation
ax2 = plt.subplot(2, 3, 2)
corr_matrix = data[numerical_features + ['approved']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, square=True)
ax2.set_title('Feature Correlation Matrix', fontweight='bold')

# Plot 3: Top 15 feature importances
ax3 = plt.subplot(2, 3, 3)
top_15 = importances.head(15)
ax3.barh(range(len(top_15)), top_15['Importance'], alpha=0.8, color='skyblue', edgecolor='black')
ax3.set_yticks(range(len(top_15)))
ax3.set_yticklabels(top_15['Feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=11)
ax3.set_title('Top 15 Feature Importance', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Model performance comparison
ax4 = plt.subplot(2, 3, 4)
exp_names = [r['Feature Set'] for r in results]
test_accs = [r['Test Acc'] for r in results]
colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax4.bar(range(len(exp_names)), test_accs, color=colors_list, alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(exp_names)))
ax4.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Test Accuracy', fontsize=11)
ax4.set_title('Model Performance by Feature Set', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontsize=9)

# Plot 5: Scaling comparison
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(data['income'], data['income_log'], alpha=0.5, label='Log Transform')
ax5.set_xlabel('Original Income', fontsize=11)
ax5.set_ylabel('Log(Income)', fontsize=11)
ax5.set_title('Log Transformation Effect', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Age distribution by group
ax6 = plt.subplot(2, 3, 6)
age_group_counts = data['age_group'].value_counts().sort_index()
ax6.bar(range(len(age_group_counts)), age_group_counts.values,
        color='coral', alpha=0.8, edgecolor='black')
ax6.set_xticks(range(len(age_group_counts)))
ax6.set_xticklabels(age_group_counts.index)
ax6.set_ylabel('Count', fontsize=11)
ax6.set_title('Age Group Distribution (Binning)', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/user/machine-learning/03-intermediate/feature_engineering_results.png', dpi=150)
print("\nVisualization saved to: 03-intermediate/feature_engineering_results.png")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING CHEAT SHEET")
print("=" * 60)
print("""
1. CATEGORICAL ENCODING:
   - Label Encoding: Ordinal data (Low < Medium < High)
   - One-Hot Encoding: Nominal data (Red, Blue, Green)
   - Target Encoding: Mean of target per category
   - Frequency Encoding: Count of each category

2. NUMERICAL TRANSFORMATIONS:
   - Log: log(x) or log1p(x) for right-skewed data
   - Square root: √x for moderate skewness
   - Box-Cox: Automatic transformation selection
   - Binning: Convert continuous to categorical

3. SCALING (Important for distance-based models!):
   - StandardScaler: mean=0, std=1 (for normal distributions)
   - MinMaxScaler: range [0,1] (for bounded features)
   - RobustScaler: Uses median/IQR (robust to outliers)

   WHEN TO SCALE:
   ✓ KNN, SVM, Neural Networks, PCA
   ✗ Tree-based models (Random Forest, Gradient Boosting)

4. INTERACTION FEATURES:
   - Multiplication: feature1 × feature2
   - Division: feature1 / feature2
   - Polynomials: x², x³, x₁x₂
   - Domain-specific: BMI = weight / height²

5. AGGREGATIONS:
   - Sum, mean, max, min across related features
   - Ratios and percentages
   - Group statistics (mean by category)

6. TIME-BASED FEATURES (for temporal data):
   - Hour, day, month, year
   - Day of week, weekend/weekday
   - Time since event
   - Rolling statistics (moving average)

7. TEXT FEATURES:
   - Length, word count
   - TF-IDF vectors
   - N-grams
   - Sentiment scores

8. FEATURE SELECTION METHODS:
   - Filter: Statistical tests (correlation, chi-square)
   - Wrapper: RFE, forward/backward selection
   - Embedded: Lasso, Random Forest importance

9. BEST PRACTICES:
   ✓ Understand your domain
   ✓ Visualize features before and after engineering
   ✓ Check for data leakage (using future information)
   ✓ Create features on train set, apply to test set
   ✓ Keep it simple: start with basic features
   ✓ Iterate: test if new features improve model

10. COMMON MISTAKES:
   ✗ Data leakage (using test data to create features)
   ✗ Not handling missing values properly
   ✗ Creating too many features (overfitting)
   ✗ Forgetting to scale/encode when necessary
   ✗ Not validating feature utility

REMEMBER: Feature engineering is iterative!
Start simple, measure impact, iterate.
""")
