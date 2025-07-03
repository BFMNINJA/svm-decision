import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 1. Load and prepare the dataset
df = pd.read_csv('breast-cancer.csv')

# Drop 'id' column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Encode target labels (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Handle missing values if any
X = X.fillna(X.mean())

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train SVM with linear and RBF kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_rbf = SVC(kernel='rbf', random_state=42)

svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)

print("Linear SVM train accuracy:", svm_linear.score(X_train_scaled, y_train))
print("RBF SVM train accuracy:", svm_rbf.score(X_train_scaled, y_train))

# 3. Visualize decision boundary using PCA (2D)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm_pca = SVC(kernel='linear', random_state=42)
svm_pca.fit(X_train_pca, y_train)

def plot_decision_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

plot_decision_boundary(svm_pca, X_train_pca, y_train, "SVM Decision Boundary (Linear, PCA 2D)")

# 4. Tune hyperparameters (C, gamma) for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Best parameters from GridSearchCV:", grid.best_params_)
print("Best cross-validated score:", grid.best_score_)

# 5. Cross-validation with best parameters
best_svm = grid.best_estimator_
cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Final test accuracy
best_svm.fit(X_train_scaled, y_train)
test_accuracy = best_svm.score(X_test_scaled, y_test)
print("Test set accuracy (best SVM):", test_accuracy)