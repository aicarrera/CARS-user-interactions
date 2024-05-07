from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import KFold
from surprise.accuracy import mae
from surprise.accuracy import rmse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from surprise.model_selection.split import train_test_split
from evaluator import *
from sklearn import metrics
import numpy as np

def predict_for_user(model,uuid):
    items=np.array(['arabiar tea', 'capuccino', 'capuccino decaffeinated', 'chocolate', 'chocolate_milk', 'cortado', 'cortado decaffeinated', 'decaffeinated', 'expresso', 'hazelnut_cappuccino', 'light_coffee', 'long_coffee', 'long_decaffeinated', 'milk_coffee', 'milk_coffee kafeinagabe', 'txokolate white'])
    ratings_est=[]
    for i in items:
        ratings_est.append(model.predict(uuid, i).est)
    ratings_est=np.array(ratings_est)
    print(items[ratings_est.argsort()[::-1]])


# Calculate sum of interactions grouped by user

# Calculate rating as sum of interactions divided by sum of interactions grouped by user
# Define the format of the data using a Reader object
reader = Reader(rating_scale=(-0.20, 1))
df = pd.read_csv('t1-doctoralStudent.csv')
df_ratings=pd.read_csv("ratings.csv")
sum_interactions = df.groupby('user')['interactions_value'].transform('sum')

min_max_scaler = MinMaxScaler()
#df["rating"]= df["sumInteractions"]/df["nroInteractions"]
#df['rating'] = df['interactions_value'] / sum_interactions
df['rating'] = df.groupby('user')['interactions_value'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print(df_ratings["value"].min(),df_ratings["value"].max(),df_ratings["value"].mean())
df=df.fillna(0)
df.loc[df.rating==0,"rating"]=0.0000001


# Load the data into the Surprise dataset format
data = Dataset.load_from_df(df_ratings[['user', 'service', 'value']], reader)
trainset, testset = train_test_split(data, test_size=0.30)

# Split the data into folds for cross-validation
kf = KFold(n_splits=3)
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNWithMeans(k=5, sim_options=sim_options)
mae_values = []
rmse_values = []
precision_values = []
recall_values = []
mrr_values=[]

# Use KNNWithMeans algorithm with cosine distance measure
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    #predict_for_user(algo,"OCUnH")
    p,r,m= evaluator(algo,df_ratings, 3)
    precision_values.append(p)
    recall_values.append(r)
    mrr_values.append(m)
    predictions = algo.test(testset)
    mae_values.append(mae(predictions))
    rmse_values.append(rmse(predictions))
    prec, rec = precision_recall_at_k(predictions, k=3, threshold=0.06249999999999996) #mayor al rating promedio
    print(prec,rec)
avg_mae = sum(mae_values) / len(mae_values)
avg_rmse = sum(rmse_values) / len(rmse_values)
avg_precision = sum(precision_values) / len(precision_values)
avg_recall = sum(recall_values) / len(recall_values)
avg_mrr = sum(mrr_values) / len(mrr_values)

#precision_values.append(precision)
#recall_values.append(recall)
#fpr, tpr, thresholds = metrics.roc_curve(testset, predictions)
#print("auc", metrics.auc(fpr, tpr))
# Compute the average MAE, precision, and recall across the k folds

print("Average MAE: ", avg_mae)
print("Average RMSE: ", avg_rmse)
print("Average precision: ", avg_precision)
print("Average recall: ", avg_recall)
print("Average mrr: ",avg_mrr)
