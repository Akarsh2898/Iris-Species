import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Load the Iris dataset
df = pd.read_csv('Iris.csv')
print("Dataset loaded successfully. Shape:", df.shape)
print(df.head())

# Prepare the data
X = df.iloc[:, 1:5].values  # Features (sepal length, sepal width, petal length, petal width)
y = df.iloc[:, 5].values    # Target (species)

# Encode the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate model performance
def evaluate_model(k, X_train, X_test, y_train, y_test):
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nResults for K = {k}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return knn, accuracy

# Function to visualize decision boundaries (optimized for efficiency)
def plot_decision_boundary(X, y, k, feature_pair=(0, 1)):
    # We'll plot the decision boundary using only 2 features at a time
    h = 0.05  # Increased step size for faster computation
    
    # Select the pair of features to visualize
    X_selected = X[:, feature_pair]
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Plot the decision boundary
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a KNN classifier with only the selected features
    knn_2d = KNeighborsClassifier(n_neighbors=k)
    knn_2d.fit(X_selected, y)
    
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    plt.xlabel(feature_names[feature_pair[0]])
    plt.ylabel(feature_names[feature_pair[1]])
    plt.title(f'Decision Boundary with k={k} for {feature_names[feature_pair[0]]} vs {feature_names[feature_pair[1]]}')
    plt.savefig(f'decision_boundary_k{k}_features_{feature_pair[0]}_{feature_pair[1]}.png')
    plt.close()

# Experiment with different values of K (reduced for efficiency)
k_values = [1, 3, 5, 7, 11]
accuracies = []

for k in k_values:
    model, accuracy = evaluate_model(k, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracies.append(accuracy)
    
    # Visualize decision boundaries for the most important feature pair only
    plot_decision_boundary(X_train_scaled, y_train, k, feature_pair=(2, 3))  # Petal length vs Petal width

# Plot accuracy vs K value
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value for KNN')
plt.xticks(k_values)
plt.grid(True)
plt.savefig('knn_accuracy_vs_k.png')
plt.close()  # Close instead of show to avoid blocking

# Find the best K value
best_k_index = np.argmax(accuracies)
best_k = k_values[best_k_index]
print(f"\nBest K value: {best_k} with accuracy: {accuracies[best_k_index]:.4f}")

print("\nAnalysis complete. Decision boundary visualizations saved as PNG files.")