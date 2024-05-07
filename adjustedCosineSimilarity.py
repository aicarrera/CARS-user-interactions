import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csr_matrix
import random

# Read CSV file and create user-item matrix
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

df = pd.read_csv('t1-doctoralStudent.csv')
df["rating"]= min_max_scaler.fit_transform(df[["interactions_value"]])
#df['rating'] = df.groupby('user')['sumInteractions'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#df['rating'] = stats.zscore(df['sumInteractions'])

# Define the z-score normalization function
#def zscore_by_user(group):
#    group['rating'] = (group['sumInteractions'] - group['sumInteractions'].mean()) / group['sumInteractions'].std()
#    return group

# Group the data by user and apply the z-score normalization function
#df = df.groupby('user').apply(zscore_by_user)


df_grouped= df.groupby(["user","service"])[["interactions_value","rating"]].sum()

df_grouped=df_grouped.reset_index()


print(df_grouped.head(100))
df_user_rating = df_grouped.pivot(index='user', columns='service', values='interactions_value')
df_item_rating = df_grouped.pivot(index='service', columns='user', values='interactions_value')
matrix_user=df_user_rating.fillna(1)

print(matrix_user)
cols=list(matrix_user.columns)
print(cols)
matrix_user=matrix_user.div(matrix_user.sum(axis=1), axis=0)
ratings_final=pd.melt(matrix_user.reset_index(),id_vars="user", value_vars=cols)
ratings_final.to_csv("ratings.csv")
print(matrix_user)
matrix_item=df_item_rating.fillna(1)

print("Sparsity %")
num_zeros = (matrix_item == 0).sum().sum()
num_entries = matrix_item.shape[0] * matrix_item.shape[1]
sparsity_pct = 100 * num_zeros / num_entries
print("Sparsity %", sparsity_pct)


# Create the heatmap with a diverging colormap
sns.heatmap(matrix_item, cmap='coolwarm', center=0, cbar=False)
plt.show()

ratingsUsers = matrix_user.reset_index()
ratingsItems=matrix_item.reset_index()
#print(ratingsUsers[["expresso","decaffeinated"]])
#print(ratingsItems[["OCUnH"]])
cosine_sim = 1-pairwise_distances(matrix_user, metric="cosine")



metric="cosine"
k=5
# This function finds k similar users given the user_id and ratings matrix M
# Note that the similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.loc[user_id, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    print('{0} most similar users for User {1} {2}:\n'.format(k, user_id, indices))
    for i in range(0, len(indices.flatten())):
        print  (   '{0}: User {1}, with similarity of {2}'.format(i, ratingsUsers.iloc[indices.flatten()[i],0], similarities[i]))

    return similarities, indices

similarities,indices = findksimilarusers("CuAgSlJwl",matrix_user, metric='cosine')


# This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, ratings, metric=metric, k=k):
    prediction = 0

    similarities, indices = findksimilarusers(user_id, ratings, metric, k)  # similar users based on cosine similarity
    mean_rating = ratings.loc[user_id, :].mean()  # to adjust for zero based indexing
    sum_wt = np.sum(similarities) - 1
    product = 1
    wtd_sum = np.zeros((ratings.shape[1],))

    for i in range(0, len(indices.flatten())):

         #   if ratingsUsers.iloc[indices.flatten()[i],0] != user_id:
                ratings_diff = ratings.iloc[indices.flatten()[i], :] - np.mean(ratings.iloc[indices.flatten()[i], :])

                product = ratings_diff * (similarities[i])

                wtd_sum = wtd_sum + product

    prediction = mean_rating + (wtd_sum / sum_wt)
    prediction=prediction.sort_values(ascending=False)
    print ('\nPredicted rating for user {0} -> item {1}'.format(user_id, prediction))

    return prediction

predict_userbased("OCUnH",matrix_user, metric='cosine')

print("Item-based Recommendation Systems")
#This function finds k similar items given the item_id and ratings matrix M

def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.loc[item_id, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    print ('{0} most similar items for item {1}:\n'.format(k,item_id))
    for i in range(0, len(indices.flatten())):
        if  ratingsItems.iloc[indices.flatten()[i],0]!= item_id:
            print('{0}: Item {1}, with similarity of {2}'.format(i, ratingsItems.iloc[indices.flatten()[i], 0], similarities[i]))
    return similarities,indices

findksimilaritems("expresso", matrix_item, metric="cosine", k=k)

# This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    similarities, indices = findksimilaritems(item_id, ratings, metric, k)  # similar users based on correlation coefficients
    sum_wt = np.sum(similarities) - 1
    product = 1
    print(similarities)
    for i in range(0, len(indices.flatten())):
        print(ratings.iloc[indices.flatten()[i]][user_id],ratingsItems.iloc[indices.flatten()[i],0])
        if ratingsItems.iloc[indices.flatten()[i],0] == item_id:
            continue;
        else:
            product = ratings.iloc[indices.flatten()[i]][user_id] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = wtd_sum / sum_wt
    print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction))

    return prediction

prediction=predict_itembased("OCUnH","expresso",matrix_item)



# This function utilizes above function to recommend items for selected approach. Recommendations are made if the predicted
# rating for an item is greater than or equal to 6, and the items has not been rated already
def recommendItem(user_id, item_id, ratings):
        approach = ['User-based CF (cosine)', 'User-based CF (correlation)', 'Item-based CF (cosine)',
                       'Item-based CF (adjusted cosine)']

        for i,a in approach.enumerate():
            print(i+" - "+a)

        opt=int(input(""))
        prediction = 0

        if (approach[opt] == 'User-based CF (cosine)'):
                    metric = 'cosine'
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
        elif (approach[opt] == 'User-based CF (correlation)'):
                    metric = 'correlation'
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
        elif (approach[opt] == 'Item-based CF (cosine)'):
           # prediction = predict_itembased(user_id, item_id, ratings)
           pass
        else:
            #prediction = predict_itembased_adjcos(user_id, item_id, ratings)
            pass

        if ratings[item_id - 1][user_id - 1] != 0:
            print
            'Item already rated'
        else:
            if prediction >= 6:
                print
                '\nItem recommended'
            else:
                print
                'Item not recommended'

