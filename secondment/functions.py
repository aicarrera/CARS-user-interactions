from surprise import Dataset, Reader, KNNWithMeans
import seaborn as sns
import random as r

import pandas as pd
from surprise.model_selection.split import train_test_split
from surprise.model_selection import KFold
from surprise.accuracy import mae
from surprise.accuracy import rmse
import sys
import numpy as np
from evaluator import evaluator, precision_recall_at_k, predict, evaluator_testset
import datetime
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


#From a rating file that has ,user,service,value  the following columns where value is the final rating
def test_knn_distances_withKFOLDS(filename,hasnegatives):
    # Load the data into the Surprise dataset format
    epsilon=sys.float_info.epsilon
    df_ratings=pd.read_csv(filename)
    #To not have issues with cosine distance I use epsilon as small value
    df_ratings.loc[df_ratings.value==0,"value"]=epsilon
    print("mean rating",df_ratings.value.mean())
    if hasnegatives:
        reader = Reader(rating_scale=(0, 1))
    else:
        reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df_ratings[['user', 'service', 'value']], reader)
    # Split the data into folds for cross-validation
    kf = KFold(n_splits=3)
    mae_values = []
    rmse_values = []
    precision_values = []
    recall_values = []
    mrr_values = []

    list_similarities=["cosine","msd", "pearson"]
    metrics=["precision","recall","mrr","mae","rmse"]
    dic_results={}
    for sim in list_similarities:
        print("Results of ", sim)
        for trainset, testset in kf.split(data):

            sim_options = {'name': sim, 'user_based': True}
            algorithm = KNNWithMeans(k=10, sim_options=sim_options)
            algorithm.fit(trainset)
            #prec, rec = precision_recall_at_k(predictions, k=3, threshold=0.01) #mayor al rating promedio
            #print(prec,rec)
            #
            results_metrics = evaluator(algorithm, df_ratings, k=1)
            precision_values.append(results_metrics[0])
            recall_values.append(results_metrics[1])
            mrr_values.append(results_metrics[2])
            predictions = algorithm.test(testset)
            mae_values.append(mae(predictions))
            rmse_values.append(rmse(predictions))

        results_metrics=[np.array(precision_values).mean(), np.array(recall_values).mean(),np.array(mrr_values).mean(),np.array(mae_values).mean(),np.array(rmse_values).mean()]
        if sim == "msd":
            sim = "euclidean"
        dic_results[sim] = {}
        for r,m in zip(results_metrics,metrics):
            dic_results[sim][m]=r


        print("-----------------------------")

    return dic_results


#From a rating file that has ,user,service,value  the following columns where value is the final rating
def test_knn_distances_splitTest(filename, times=20, kval=3, hasnegatives=True,mean_rating=0.01):
    # Load the data into the Surprise dataset format
    epsilon=sys.float_info.epsilon
    df_ratings=pd.read_csv(filename)
    #print(df_ratings.head())
    #print(df_ratings.dtypes)
    #print(df_ratings.columns[df_ratings.isna().any()])
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    #To not have issues with cosine distance I use epsilon as small value
    df_ratings.loc[df_ratings.value==0,"value"]=epsilon
    #mean_rating= df_ratings.value.mean()
    print("mean rating",df_ratings.value.mean(), mean_rating)
    if hasnegatives:
        reader = Reader(rating_scale=(-1, 1))
    else:
        reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df_ratings[['user', 'service', 'value']], reader)
    # Split the data into folds for cross-validation
    list_similarities=["cosine","msd", "pearson"]
    dic_results={}
    f=open("resultsIDEKO/results_{}_{}_k{}_{}".format(filename.split("/")[-1].split(".")[0],timestamp,kval,hasnegatives),"w")
    f.write("distance,k,i,precision,recall,mrr,mae,rmse\n")
    for i in range(times):
        seed = r.randint(1, 100)
        trainset, testset = train_test_split(data, test_size=0.30)
        #print(testset)
        for sim in list_similarities:
            print("Results of ", sim)
            sim_options = {'name': sim, 'user_based': True}
            algorithm = KNNWithMeans(k=5, sim_options=sim_options)
            algorithm.fit(trainset)
            #prec, rec = precision_recall_at_k(predictions, k=3, threshold=0.01) #mayor al rating promedio
            #print(prec,rec)
            #
            precision_val,recall_val,mrr_val = evaluator_testset(algorithm,testset, df_ratings, kval, mean_rating,seed)
            #precision_val, recall_val, mrr_val = evaluator(algorithm,df_ratings, kval)

            predictions = algorithm.test(testset)
            if sim == "msd":
                sim = "euclidean"
            f.write("{},{},{},{},{},{},{},{}\n".format(sim,kval,i,precision_val, recall_val,mrr_val,mae(predictions),rmse(predictions)))
    print("-----------------------------")
    f.close()
    return dic_results


def predict_model_foruser(filename,k,uuid):
    # Load the data into the Surprise dataset format
    epsilon = sys.float_info.epsilon
    df_ratings = pd.read_csv(filename)
    print(df_ratings.head())
    print(df_ratings.dtypes)
    print(df_ratings.columns[df_ratings.isna().any()])

    # To not have issues with cosine distance I use epsilon as small value
    df_ratings.loc[df_ratings.value == 0, "value"] = epsilon
    print("mean rating", df_ratings.value.mean())

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df_ratings[['user', 'service', 'value']], reader)
    list_similarities=["cosine","msd", "pearson"]
    trainset, testset = train_test_split(data, test_size=0.20)

    for sim in list_similarities:
        sim_options = {'name': sim, 'user_based': True}
        algorithm = KNNWithMeans(k=5, sim_options=sim_options)
        algorithm.fit(trainset)

        predict(algorithm, df_ratings, k, uuid)

def boxplot_results_context(filenames, metric,folder="results/"):
    print("boxplot")
    all_results = pd.DataFrame()
    for f in filenames:
        # Load the results into a DataFrame
        results_df = pd.read_csv(folder+f)
        results_df["file"]=f
        all_results = pd.concat([all_results, results_df], ignore_index=True)
    grouped = all_results.groupby(['file', 'distance','k']).mean()[['precision', 'recall', 'mrr', 'mae', 'rmse']]
    grouped=grouped.reset_index()
    grouped = grouped.groupby('k')
    # Create a subplot for each k value
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    row = 0

    # Iterate through the groups and create a separate image for each group
    for name, group in grouped:
        # Use Seaborn to create the box plot
        print(name, group)
        sns.boxplot(ax=axes[0,row], data=group, x='distance', y=metric,palette=sns.color_palette("YlGnBu", n_colors=5))
        x=sns.barplot(ax=axes[1, row], data=group, x='distance', y=metric, palette=sns.color_palette("YlGnBu", n_colors=5))
        for i in x.containers:
            x.bar_label(i, )
        # Set the axis labels and title
        axes[0,row].set_xlabel('Distance Metric')
        axes[0,row].set_title('{} for K={}'.format(metric.upper(), name))

        row += 1


def lineplot_results(filenames, metric, folder="results/"):
    all_results = pd.DataFrame()
    # Load the results into a DataFrame
    for f in filenames:
        results_df = pd.read_csv(folder+f)
        all_results = pd.concat([all_results, results_df], ignore_index=True)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group the data by distance and k
    grouped = all_results[["k","distance",metric]].groupby(["k","distance"]).mean().reset_index()
    grouped = grouped.sort_values(by='k', ascending=False)
    grouped['k'] = grouped['k'].astype('str').astype('category')

   # print(grouped.head(10))

    #grouped_wide = grouped.pivot("k", "distance", metric)

    sns.lineplot(x="k", data=grouped, y=metric, hue='distance', style='distance', markers=True, dashes=False)

    # Show the plot
    plt.show()


def boxplot_results(filenames, metric,folder="results/"):
    all_results = pd.DataFrame()
    for f in filenames:
        # Load the results into a DataFrame
        results_df = pd.read_csv(folder+f)
        all_results = pd.concat([all_results, results_df], ignore_index=True)
        # Group the data by k

    grouped = all_results.groupby('k')
    # Set up the figure and axis
    fig, axes = plt.subplots(2,3,figsize=(14, 8))
    row = 0

    # Iterate through the groups and create a separate image for each group
    for name, group in grouped:
        # Use Seaborn to create the box plot
        sns.boxplot(ax=axes[0,row], data=group, x='distance', y=metric, palette=sns.color_palette("YlGnBu", n_colors=5) )
        x=sns.barplot(ax=axes[1, row], data=group, x='distance', y=metric, estimator=np.mean, palette=sns.color_palette("YlGnBu", n_colors=5))
        for i in x.containers:
            x.bar_label(i, )
        # Set the axis labels and title
        axes[0,row].set_ylabel("")
        axes[0,row].set_xlabel('Distance Metric')
        axes[0,row].set_title('{}@K={}'.format(metric.upper(), name))

        row+=1
    # Display the plot


#predict_model_foruser("filesToProcess/ratings_t1-doctoralstudent_False.csv",5,"OCUnH")
def naive_recommender(filename, testing_size):
    # Read the csv file
    df = pd.read_csv(filename)
    users=df["user"].unique()
    df=df[df["value"]!=0]
    # Split filtered_df into two dataframes based on the given percentage porc
    filtered_size  = df.shape[0]
    split_index = int(filtered_size  * testing_size)
    test_df = df.sample(n=split_index)
    training_df = df.drop(test_df.index)

    # Group by user and get the maximum id for each user
    grouped_df = training_df.groupby("user")["id"].max().reset_index()
    grouped_df.rename(columns={"id": "max_id"}, inplace=True)

    # Merge the grouped_df with the original df based on the user column
    merged_df = pd.merge(training_df, grouped_df, on="user")

    # Filter the rows in the merged_df where the id equals the max_id per user
    model_df = merged_df[merged_df["id"] == merged_df["max_id"]]

    return test_df, training_df, model_df, users
'''
def naive_recommender(filename, testing_size, k):
    # Read the csv file
    df = pd.read_csv(filename)
    users=df["user"].unique()
    df=df[df["value"]!=0]
    # Split filtered_df into two dataframes based on the given percentage porc
    filtered_size  = df.shape[0]
    split_index = int(filtered_size  * testing_size)
    test_df = df.sample(n=split_index)
    training_df = df.drop(test_df.index)

    # Group by user and get the top k maximum ids for each user
    grouped_df = training_df.sort_values(by=["user", "id"], ascending=False).groupby("user").head(k)

    return test_df, training_df, grouped_df, users

test_df, training_df, model_df, users=naive_recommender("filesToProcess/query-result.csv",0.20,3)
print(model_df)

'''

'''
import os
filenames=[]
for filename in os.listdir("results/"):
    if os.path.isfile(os.path.join("results/", filename)):
        if "True" in filename:
            filenames.append(filename)
print(filenames)
if len(filenames)>0:
    boxplot_results(filenames,"precision","results/")
'''

def table(filenames,filenames2, metric, folderWithContext="results/", folderWithoutContext=""):
    all_results = pd.DataFrame()
    for f in filenames:
        # Load the results into a DataFrame
        results_df = pd.read_csv(folderWithContext + f)
        results_df["file"] = f
        all_results = pd.concat([all_results, results_df], ignore_index=True)
    all_results["context"]=True

    all_results2 = pd.DataFrame()
    for f in filenames2:
        # Load the results into a DataFrame
        results_df = pd.read_csv(folderWithoutContext+f)
        all_results2 = pd.concat([all_results2, results_df], ignore_index=True)
    all_results2["context"]=False
    all_results = pd.concat([all_results, all_results2], ignore_index=True)

    grouped = all_results.groupby(['context', 'distance', 'k']).mean()[['precision']]
    grouped = grouped.reset_index()
    df_pivot = grouped.pivot(index='k', columns=["context",'distance'], values='precision')
    print(df_pivot)
    print(df_pivot.to_latex())
import os
filenames=[]
for filename in os.listdir("resultsIDEKO/16012024WithContextKindofHappy/"):
    if os.path.isfile(os.path.join("resultsIDEKO/16012024WithContextKindofHappy/", filename)):
        if "False" in filename:
            filenames.append(filename)
filenames2=[]
for filename in os.listdir("resultsIDEKO/"):
    if os.path.isfile(os.path.join("resultsIDEKO/", filename)):
        if "False" in filename:
            filenames2.append(filename)
if len(filenames)>0:
    table(filenames,filenames2,"precision","resultsIDEKO/16012024WithContextKindofHappy/","resultsIDEKO/")
