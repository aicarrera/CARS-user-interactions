import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from secondment.functions import test_knn_distances_withKFOLDS, test_knn_distances_splitTest, boxplot_results_context, \
    boxplot_results

def processFileImplicitExplicit(filename):
    df = pd.read_csv('filesToProcess/'+filename)
    dfInteractions = df.loc[df.type == "INTERACTION", :]
    dfRatings = df.loc[df.type == "RATING", :]
    weight_implicit=1
    if len(dfRatings)>0:
        weight_explicit= 0.3
        weight_implicit-=weight_explicit
    else:
        weight_explicit= 0

    #IMPLICIT - INTERACTIONS
    grouped_interactions = dfInteractions.groupby(['user', 'service']).agg({'value': ['sum', 'count']})
    # remove one level and reset index so rows repeat
    grouped_interactions.columns = grouped_interactions.columns.droplevel(level=0)
    grouped_interactions = grouped_interactions.reset_index()
    countInteractionsUser = grouped_interactions[["user", "sum"]].groupby("user").sum().reset_index()
    matrix_user_rating = grouped_interactions.pivot(index='user', columns='service', values='sum').fillna(0)
    matrix_implicit_rating = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)

    #EXPLICIT- RATINGS
    grouped_ratings = dfRatings.groupby(['user', 'service']).agg({'value': ['mean']})
    grouped_ratings.columns = grouped_ratings.columns.droplevel(level=0)
    grouped_ratings = grouped_ratings.reset_index()
    # Round the mean
    grouped_ratings['mean'] = grouped_ratings['mean'].round()
    # Normalize the values
    scaler = MinMaxScaler(feature_range=(-0.01, 1))
    grouped_ratings['value_normalized'] = scaler.fit_transform(grouped_ratings[['mean']])
    matrix_explicit_rating = grouped_ratings.pivot(index='user', columns='service', values='value_normalized').fillna(0)
    # reindex the matrices to ensure they have the same columns and rows
    index_labels = matrix_implicit_rating.index.union(matrix_explicit_rating.index)
    column_labels = matrix_implicit_rating.columns.union(matrix_explicit_rating.columns)
    matrix_implicit_rating = matrix_implicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)
    matrix_explicit_rating = matrix_explicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)
    # Combine the implicit and explicit feedback matrices
    combined_matrix = (weight_explicit * matrix_explicit_rating) + (weight_implicit * matrix_implicit_rating)

    cols = list(combined_matrix.columns)
    ratings_final = pd.melt(combined_matrix.reset_index(), id_vars="user", value_vars=cols)
    ratings_final.to_csv("IDEKOfilesToProcess/ratings" + str("combined") + ".csv")



def processFile(filename, change=False):
    df = pd.read_csv('IDEKOfilesToProcess/'+filename)
    if change:
        df.loc[df.value == 0, "value"] = -1
    grouped_df = df.groupby(['user', 'service']).agg({'value': ['sum', 'count']})
    #remove one level and reset index so rows repeat
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    #Grouped by users
    countInteractionsUser=grouped_df[["user","sum"]].groupby("user").sum()
    countInteractionsUser=countInteractionsUser.reset_index()
    #
    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='sum')
    matrix_user_rating = matrix_user_rating.fillna(0)

    # Divide countInteractionsUser by matrix_user_rating
    matrix_user_interaction_ratio = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)
    cols=list(matrix_user_interaction_ratio.columns)
    ratings_final=pd.melt(matrix_user_interaction_ratio.reset_index(),id_vars="user", value_vars=cols)
    ratings_final.to_csv("IDEKOfilesToProcess/ratings"+str(change)+".csv")

def processFileGroup(filename, change=False):
    df = pd.read_csv('IDEKOfilesToProcess/'+filename)
    if change:
        df.loc[df.value == 0, "value"] = -1
    grouped_df = df.groupby(['user','task', 'service']).agg({'value': ['sum', 'count']})
    #remove one level and reset index so rows repeat
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    #Grouped by users
    countInteractionsUser=grouped_df[["user","task","sum"]].groupby(["user","task"]).sum()
    countInteractionsUser=countInteractionsUser.reset_index()
    print(countInteractionsUser)

    grouped_df = pd.merge(grouped_df, countInteractionsUser, on=['user', 'task'], suffixes=('', '_total'))
    grouped_df['value_ratio'] = grouped_df['sum'] / grouped_df['sum_total']

    print(grouped_df)

    #
    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='value_ratio')
    matrix_user_rating=matrix_user_rating.fillna(0)

    # Divide countInteractionsUser by matrix_user_rating
    #matrix_user_interaction_ratio = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)
    cols=list(matrix_user_rating.columns)
    ratings_final=pd.melt(matrix_user_rating.reset_index(),id_vars="user", value_vars=cols)
    ratings_final.to_csv("IDEKOfilesToProcess/ratings"+str(change)+".csv")



def processFileMinMaxScaler(filename, change=False):
    df = pd.read_csv('IDEKOfilesToProcess/'+filename)
    if change:
        df.loc[df.value == 0, "value"] = -1
    grouped_df = df.groupby(['user', 'service']).agg({'value': ['sum', 'count']})
    #remove one level and reset index so rows repeat
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    countInteractionsUser=grouped_df[["user","count"]].groupby("user").sum()
    countInteractionsUser=countInteractionsUser.reset_index()

    #The reason for using a default rating of 0.5 for users with only one row is to avoid giving them the highest rating of 1 or the lowest rating of 0. This is because the min-max scaler method is based on the range of values in the data, and for a single data point, the range is zero, leading to a division by zero error.
    #By setting the default rating to 0.5, we are assigning a neutral rating to users with only one row, which is halfway between the minimum and maximum possible rating values of 0 and 1. This approach is often used in cases where the data is sparse or where there are missing values, to avoid overestimating or underestimating the rating for a particular user.
    grouped_df['value'] = grouped_df.groupby('user')['sum'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    grouped_df.loc[grouped_df.value == 0, "value"] = 0.001
    grouped_df['value']=grouped_df['value'].fillna(0.5)

    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='value')
    matrix_user_rating = matrix_user_rating.fillna(0)
    cols=list(matrix_user_rating.columns)
    ratings_final = pd.melt(matrix_user_rating.reset_index(), id_vars="user", value_vars=cols)
    ratings_final.to_csv("IDEKOfilesToProcess/ratingsMinMax" + str(change) + ".csv")


#processFileMinMaxScaler("query-result.csv",True)
def processFileWithContextImplicitExplicit(filename,contextList, weight_explicit):
    df = pd.read_csv('IDEKOfilesToProcess/' + filename)
    rows_to_add= df.groupby(["id","user","subservice_name","value","startdate","service_name"]).count().reset_index()
    selected_columns = ['id', 'user', 'subservice_name', 'value', 'startdate',"service_name"]
    rows_to_add= rows_to_add[selected_columns]
    rows_to_add["context_name"]="task"
    rows_to_add.rename(columns={'service_name': 'contextvalue'}, inplace=True)
    # Extract relevant columns
    selected_columns = ['id', 'user', 'subservice_name', 'value', 'startdate',  'contextvalue', 'context_name']
    df_selected = df[selected_columns]
    # Concatenate rows_to_add to df_selected
    result_df = pd.concat([df_selected, rows_to_add[selected_columns]])
    result_df.to_csv("IDEKOfilesToProcess/preprocessed_data.csv")
    result_df["type"]="INTERACTION"
    df = result_df[result_df['contextvalue'].isin(contextList)]
    dfInteractions = df.loc[df.type == "INTERACTION", :]
    dfRatings = df.loc[df.type == "RATING", :]

    weight_implicit = 1 - weight_explicit
    if len(dfRatings) > 0:
        weight_implicit -= weight_explicit
    else:
        weight_explicit = 0

    #IMPLICIT RATINGSw
    grouped_df = dfInteractions.groupby(['user', 'subservice_name']).agg({'value': ['sum']})
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    countInteractionsUser = grouped_df[grouped_df['sum'] > 0][["user", "sum"]].groupby("user").sum().reset_index()
    matrix_user_rating = grouped_df.pivot(index='user', columns='subservice_name', values='sum').fillna(0)
    # Divide countInteractionsUser by matrix_user_rating
    matrix_implicit_rating = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)

    #EXPLICIT RATINGS
    if len(dfRatings) > 0:
        grouped_ratings = dfRatings.groupby(['user', 'subservice_name']).agg({'value': ['mean']})
        grouped_ratings.columns = grouped_ratings.columns.droplevel(level=0)
        grouped_ratings = grouped_ratings.reset_index()
        # Round the mean
        grouped_ratings['mean'] = grouped_ratings['mean'].round()
        # Normalize the values
        scaler = MinMaxScaler(feature_range=(-0.01, 1))
        grouped_ratings['value_normalized'] = scaler.fit_transform(grouped_ratings[['mean']])
        matrix_explicit_rating = grouped_ratings.pivot(index='user', columns='subservice_name', values='value_normalized').fillna(0)
        # reindex the matrices to ensure they have the same columns and rows

        index_labels = matrix_implicit_rating.index.union(matrix_explicit_rating.index)
        column_labels = matrix_implicit_rating.columns.union(matrix_explicit_rating.columns)
        matrix_explicit_rating = matrix_explicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)

    index_labels = matrix_implicit_rating.index
    column_labels = matrix_implicit_rating.columns
    matrix_implicit_rating = matrix_implicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)
    # Combine the implicit and explicit feedback matrices
    if len(dfRatings) > 0:
        combined_matrix = (weight_explicit * matrix_explicit_rating) + (weight_implicit * matrix_implicit_rating)
    else:
        combined_matrix = (weight_implicit * matrix_implicit_rating)

    cols = list(combined_matrix.columns)
    ratings_final = pd.melt(combined_matrix.reset_index(), id_vars="user", value_vars=cols)
    ratings_final.rename(columns={'subservice_name': 'service'}, inplace=True)

    ratings_final.to_csv("IDEKOfilesToProcess/ratings_" +"-".join(contextList).replace(" ","") +"_" + str("combined") + ".csv")
    return "IDEKOfilesToProcess/ratings_" +"-".join(contextList).replace(" ","") +"_" + str("combined") + ".csv"


def processFileWithContextImplicitExplicit_exclusive_inclusive(filename,contextList, weight_explicit):
    df = pd.read_csv('IDEKOfilesToProcess/' + filename)
    print(contextList[1:])
    df = df.loc[df.service_name == contextList[0], :]
    df = df[df['contextvalue'].isin(contextList[1:])]
    df["type"] = "INTERACTION"

    dfInteractions = df.loc[df.type == "INTERACTION", :]
    # Rename the column
    dfInteractions = dfInteractions.rename(columns={'subservice_name': 'service'})

    dfRatings = df.loc[df.type == "RATING", :]
    weight_implicit = 1 - weight_explicit
    if len(dfRatings) > 0:
        weight_implicit -= weight_explicit
    else:
        weight_explicit = 0
    # IMPLICIT RATINGSw
    grouped_df = dfInteractions.groupby(['user', 'service']).agg({'value': ['sum']})
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    countInteractionsUser = grouped_df[grouped_df['sum'] > 0][["user", "sum"]].groupby("user").sum().reset_index()
    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='sum').fillna(0)
    # Divide countInteractionsUser by matrix_user_rating
    matrix_implicit_rating = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)
    # EXPLICIT RATINGS
    if len(dfRatings) > 0:
        # EXPLICIT RATINGS
        grouped_ratings = dfRatings.groupby(['user', 'service']).agg({'value': ['mean']})
        grouped_ratings.columns = grouped_ratings.columns.droplevel(level=0)
        grouped_ratings = grouped_ratings.reset_index()
        # Round the mean
        grouped_ratings['mean'] = grouped_ratings['mean'].round()
        # Normalize the values
        scaler = MinMaxScaler(feature_range=(-0.01, 1))
        grouped_ratings['value_normalized'] = scaler.fit_transform(grouped_ratings[['mean']])
        matrix_explicit_rating = grouped_ratings.pivot(index='user', columns='service', values='value_normalized').fillna(0)
        # reindex the matrices to ensure they have the same columns and rows
        index_labels = matrix_implicit_rating.index.union(matrix_explicit_rating.index)
        column_labels = matrix_implicit_rating.columns.union(matrix_explicit_rating.columns)
        matrix_implicit_rating = matrix_implicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)
        matrix_explicit_rating = matrix_explicit_rating.reindex(index=index_labels, columns=column_labels, fill_value=0)

    # Combine the implicit and explicit feedback matrices
    if len(dfRatings) > 0:
        combined_matrix = (weight_explicit * matrix_explicit_rating) + (weight_implicit * matrix_implicit_rating)
    else:
        combined_matrix = (weight_implicit * matrix_implicit_rating)

    #**************************************************************************
    # Create a file path based on your naming convention
    file_path = "IDEKOfilesToProcess/ratings_" + "-".join(contextList[:-1]).replace(" ", "") + "_" + str(
            "combined") + ".csv"

    cols = list(combined_matrix.columns)
    if os.path.exists(file_path):
            # Append the new data to the existing data
            ratings_final = pd.melt(combined_matrix.reset_index(), id_vars="user", value_vars=cols)
            # Save the combined data to the same file in append mode
            ratings_final.to_csv(file_path, mode='a', header=False)
    else:
            # If the file doesn't exist, save the data as usual
            ratings_final = pd.melt(combined_matrix.reset_index(), id_vars="user", value_vars=cols)
            ratings_final.to_csv(file_path)
    return file_path
#  ratings_final.to_csv("IDEKOfilesToProcess/ratings_" + "-".join(contextList[:-1]).replace(" ", "") + "_" + str("combined") + ".csv")
#  return "IDEKOfilesToProcess/ratings_" + "-".join(contextList[:-1]).replace(" ", "") + "_" + str("combined") + ".csv"

def processFileWithContext(filename, contextList, negatives=False):
    df = pd.read_csv('IDEKOfilesToProcess/'+filename)
    df = df[df['contextValue'].isin(contextList)]
    if negatives:
        df.loc[df.value == 0, "value"] = -1
    grouped_df = df.groupby(['user', 'service']).agg({'value': ['sum']})
    #remove one level and reset index so rows repeat
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    grouped_df.to_csv("IDEKOfilesToProcess/review_" +"-".join(contextList) +"_" + str(negatives) + ".csv")
    #Grouped by users
   # countInteractionsUser=grouped_df[["user","sum"]].groupby("user").sum()
    countInteractionsUser = grouped_df[grouped_df['sum'] > 0][["user", "sum"]].groupby("user").sum().reset_index()

    countInteractionsUser=countInteractionsUser.reset_index()
    #
    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='sum')
    matrix_user_rating = matrix_user_rating.fillna(0)
    # Divide countInteractionsUser by matrix_user_rating
    matrix_user_interaction_ratio = matrix_user_rating.div(countInteractionsUser.set_index('user')['sum'], axis=0)
    cols=list(matrix_user_interaction_ratio.columns)
    ratings_final=pd.melt(matrix_user_interaction_ratio.reset_index(),id_vars="user", value_vars=cols)
    ratings_final.to_csv("IDEKOfilesToProcess/ratings_" +"-".join(contextList).replace(" ","") +"_" + str(negatives) + ".csv")
    return "IDEKOfilesToProcess/ratings_" +"-".join(contextList).replace(" ","") +"_" + str(negatives) + ".csv"

#-------------------------------------------------------------------------------------------------------------------
def processFileWithContextMinMaxScaler(filename, contextList):
    df = pd.read_csv('IDEKOfilesToProcess/' + filename)
    df = df[df['contextValue'].isin(contextList)]

    df.loc[df.value == 0, "value"] = -0.5
    grouped_df = df.groupby(['user', 'service']).agg({'value': ['sum']})
    # remove one level and reset index so rows repeat
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()
    matrix_user_rating = grouped_df.pivot(index='user', columns='service', values='sum')
    matrix_user_rating = matrix_user_rating.fillna(0)
    # Divide countInteractionsUser by matrix_user_rating
    cols = list(matrix_user_rating.columns)
    ratings_final = pd.melt(matrix_user_rating.reset_index(), id_vars="user", value_vars=cols)

    ratings_final['value'] = ratings_final.groupby('user')['value'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    #ratings_final.to_csv("filesToProcess/review_" + "-".join(contextList)  + ".csv")

    ratings_final.to_csv("IDEKOfilesToProcess/ratings_" + "-".join(contextList).replace(" ", "") + ".csv")

    return "IDEKOfilesToProcess/ratings_" + "-".join(contextList).replace(" ", "") + ".csv"


def process_files_datasetfolder(filename):
    df = pd.read_csv(filename)
    df["rating"]=df.sumInteractions/df.nro_interactions
    #Matrix user-service rating
    matrix_user_rating = df.pivot(index='user', columns='service', values='rating')
    matrix_user_rating=matrix_user_rating.fillna(0)
    #Saving ratings to use in Surprise
    cols=list(matrix_user_rating.columns)
    ratings_final=pd.melt(matrix_user_rating.reset_index(),id_vars="user", value_vars=cols)
    ratings_final.to_csv("datasets/ratings.csv")#

testingContext=False
kfolds=False
testingwithoutcontext=True
traintestSplit=True
for i in range(1,4):
    k=i
    NEGATIVE=False
    if testingContext :
        turns=["t1","t2"]
        roles=["GrindingResearcher","GeneralResearcher"]
        services=["signal_monitoring","indicators-selection","schema_monitoring","transformation_execution","machine_selection"]
        files=[]
        for s in services:
            for r in roles:
                for t in turns :
                #for d in days:
                #files.append(processFileWithContext("query-result-allContext_old.csv",[t,r],negatives=NEGATIVE))
                #este fue el ultimo files.append(processFileWithContextMinMaxScaler("query-result-allContext.csv", [t,r]))
                    #files.append(processFileWithContextImplicitExplicit("_resultType_allContexts.csv", [r,s],0))
                    files.append(processFileWithContextImplicitExplicit_exclusive_inclusive("_resultType_allContexts.csv", [s, r,t], 0))

        results=[]

        for f in files:
            print("File", f)

            context_results = test_knn_distances_splitTest(f, kval=k, times=5, hasnegatives=NEGATIVE)
            results.append({f:context_results})
        print(results)

    elif testingwithoutcontext:
        if kfolds:
            print("**Without negatives")
            processFile("query-result-withoutContext.csv")
            general_results=test_knn_distances_withKFOLDS("IDEKOfilesToProcess/ratingsFalse.csv", False)
            print(general_results)
            print("**With negatives")
            processFile("query-result-withoutContext.csv",True)
            general_results=test_knn_distances_withKFOLDS("IDEKOfilesToProcess/ratingsTrue.csv", True)
            print(general_results)

        if traintestSplit:
            print("**Without negatives")
            processFileGroup("queryRatingwithTask.csv")
            #processFile("query-result-withoutContext.csv")
            general_results=test_knn_distances_splitTest("IDEKOfilesToProcess/ratingsFalse.csv",hasnegatives=False,kval=k, times=20, mean_rating=0.25)
            print(general_results)
            #print("**With negatives")
            #processFileImplicitExplicit("_resultType_withoutContext.csv")
            #general_results=test_knn_distances_splitTest("filesToProcess/ratingscombined.csv",hasnegatives=True,kval=k)
            #print(general_results)

    #general_results=test_knn_distances_splitTest("filesToProcess/ratingsMinMaxTrue.csv",hasnegatives=False,kval=k)
    #print(general_results)






'''
#Check this later since i am not sure this files are correct
 #Reading all_interactions.csv that contains values without filtering context
general_results=test_knn_distances("datasets/all_interactions.csv")
print(general_results)

files=['t1-doctoralstudent.csv','t2-doctoralstudent.csv']

for f in files:
    print(f)
    context_results = test_knn_distances(f)
    print(context_results)
'''