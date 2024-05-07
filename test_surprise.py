import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load iris dataset and split into training and testing sets
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Calculate inverse covariance matrix of training data
covariance_matrix = np.cov(X_train.T)
inv_cov = np.linalg.inv(covariance_matrix)

# Initialize KNN with Mahalanobis distance metric
knn = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis', metric_params={'V': inv_cov})

# Fit KNN model to training data
knn.fit(X_train, y_train)

# Predict class labels of test data
y_pred = knn.predict(X_test)

# Print accuracy score of KNN model
print("Accuracy:", knn.score(X_test, y_test))
