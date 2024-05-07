import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from scipy.stats import pearsonr
import sys

def pearson_distance(x, y):
    mask = np.logical_or(np.isnan(x), np.isnan(y))
    x_masked = np.ma.masked_array(x, mask=mask)
    y_masked = np.ma.masked_array(y, mask=mask)
    if np.all(x_masked == x_masked[0]) or np.all(y_masked == y_masked[0]):
        return np.inf
    corr, _ = pearsonr(x_masked, y_masked)
    return 1 - corr

# Load dataset into a Pandas dataframe
df = pd.read_csv('datasets/ratings.csv')

# Divide the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create pivot table for training set
pivot_table_train = train_df.pivot(index='user', columns='service', values='value')

# Replace missing values with zeros
epsilon=sys.float_info.epsilon

pivot_table_train = pivot_table_train.fillna(epsilon)

# Calculate inverse covariance matrix of pivot table
covariance_matrix = np.cov(pivot_table_train.T)
inv_cov = np.linalg.inv(covariance_matrix)


distances_dicc = {"mahalanobis":{'VI': inv_cov},"euclidean":None, pearson_distance:None, "cosine":None}

for d in distances_dicc:
    # Initialize KNN model with Mahalanobis distance metric
    knn = NearestNeighbors(metric=d, metric_params=distances_dicc[d])

    # Fit KNN model to pivot table
    knn.fit(pivot_table_train)

    # Find k-nearest neighbors for each user
    k = 10
    distances, indices = knn.kneighbors(pivot_table_train, n_neighbors=k+1)

    # Make recommendations for each user
    recommendations = {}
    for i, row in enumerate(pivot_table_train.index):
        neighbor_indices = indices[i][1:]
        neighbor_ratings = pivot_table_train.iloc[neighbor_indices]

        mean_ratings = neighbor_ratings.mean()
        recommended_services = mean_ratings.sort_values(ascending=False).index.tolist()
        recommendations[row] = recommended_services

    # Print recommendations for a specific user
    # print(recommendations)
    # Define the value of k
    k = 3

    # Initialize variables to keep track of total recommendations and correct recommendations
    total_recommendations = 0
    correct_recommendations = 0

    # Loop over each user and their recommended services
    for user, recommended in recommendations.items():
        # Select the first k recommendations for each user
        recommended_k = recommended[:k]

        # Get the actual ratings of the user
        actual = pivot_table_train.loc[user]

        # Only consider the actual ratings that are not missing (i.e., not equal to 0)
        actual = actual[actual != 0]

        # Count the total number of recommendations and correct recommendations
        total_recommendations += len(recommended_k)
        correct_recommendations += len(set(recommended_k).intersection(set(actual.index)))

    # Calculate precision@k
    precision_at_k = correct_recommendations / total_recommendations
    print(f"Precision@{k} = {precision_at_k:.2f}")


