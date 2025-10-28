import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the Iris dataset
print("Loading Iris dataset...")
df = pd.read_csv('Iris.csv')

# Display basic information about the dataset
print("\n=== Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nTarget variable distribution:")
print(df['Species'].value_counts())

# Separate features and target
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_columns].copy()
y = df['Species'].copy()

print(f"\n=== Feature Statistics (Before Normalization) ===")
print(X.describe())

# Visualize original data distribution
plt.figure(figsize=(15, 10))

# Original features distribution
plt.subplot(2, 3, 1)
X.boxplot()
plt.title('Original Features Distribution')
plt.xticks(rotation=45)

# Apply different normalization techniques
print(f"\n=== Applying Normalization Techniques ===")

# 1. Standard Scaler (Z-score normalization)
standard_scaler = StandardScaler()
X_standard = pd.DataFrame(
    standard_scaler.fit_transform(X),
    columns=feature_columns
)

print("1. StandardScaler applied (mean=0, std=1)")
print("   Formula: (x - mean) / std")
print(f"   Mean: {X_standard.mean().round(6).values}")
print(f"   Std:  {X_standard.std().round(6).values}")

# 2. Min-Max Scaler (normalization to 0-1 range)
minmax_scaler = MinMaxScaler()
X_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(X),
    columns=feature_columns
)

print(f"\n2. MinMaxScaler applied (range 0-1)")
print("   Formula: (x - min) / (max - min)")
print(f"   Min: {X_minmax.min().round(6).values}")
print(f"   Max: {X_minmax.max().round(6).values}")

# 3. Robust Scaler (uses median and IQR)
robust_scaler = RobustScaler()
X_robust = pd.DataFrame(
    robust_scaler.fit_transform(X),
    columns=feature_columns
)

print(f"\n3. RobustScaler applied (uses median and IQR)")
print("   Formula: (x - median) / IQR")
print(f"   Median: {X_robust.median().round(6).values}")

# Visualize normalized data
plt.subplot(2, 3, 2)
X_standard.boxplot()
plt.title('StandardScaler Normalized')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
X_minmax.boxplot()
plt.title('MinMaxScaler Normalized')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
X_robust.boxplot()
plt.title('RobustScaler Normalized')
plt.xticks(rotation=45)

# Create correlation heatmap
plt.subplot(2, 3, 5)
correlation = X.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')

# Feature distribution by species
plt.subplot(2, 3, 6)
for species in df['Species'].unique():
    species_data = df[df['Species'] == species]
    plt.scatter(species_data['SepalLengthCm'], species_data['PetalLengthCm'], 
               label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Species Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('iris_normalization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare the effect of normalization on a simple classifier
print(f"\n=== Classification Performance Comparison ===")

def evaluate_classifier(X_data, y_data, data_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\n{data_name}:")
    print(f"  Training Accuracy: {train_score:.4f}")
    print(f"  Testing Accuracy:  {test_score:.4f}")
    
    return test_score

# Test different normalized versions
scores = {}
scores['Original'] = evaluate_classifier(X, y, "Original Data")
scores['StandardScaler'] = evaluate_classifier(X_standard, y, "StandardScaler")
scores['MinMaxScaler'] = evaluate_classifier(X_minmax, y, "MinMaxScaler") 
scores['RobustScaler'] = evaluate_classifier(X_robust, y, "RobustScaler")

# Summary of normalization effects
print(f"\n=== Summary ===")
print("Normalization Technique Comparison:")
for method, score in scores.items():
    print(f"  {method:<15}: {score:.4f}")

print(f"\nNormalization Benefits:")
print("1. StandardScaler: Best for normally distributed data, centers around mean=0")
print("2. MinMaxScaler: Best when you need bounded features [0,1], preserves relationships")
print("3. RobustScaler: Best when data has outliers, uses median instead of mean")
print(f"\nFor Iris dataset: All methods perform similarly due to the clean, well-distributed data.")
print(f"StandardScaler is typically recommended for neural networks and SVM.")
print(f"MinMaxScaler is good for algorithms sensitive to feature scales like KNN.")

# Save normalized datasets
X_standard.to_csv('iris_standardscaler.csv', index=False)
X_minmax.to_csv('iris_minmaxscaler.csv', index=False)
X_robust.to_csv('iris_robustscaler.csv', index=False)

print(f"\nNormalized datasets saved:")
print("- iris_standardscaler.csv")
print("- iris_minmaxscaler.csv") 
print("- iris_robustscaler.csv")
