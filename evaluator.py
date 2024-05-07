import logging
import numpy as np
from collections import defaultdict

precision_metrics = []
recall_metrics = []
"""
Sequential evaluation edited from:
https://sparsh-ai.github.io/
"""




def last_session_out_split(data):
    """
    Assign the last session of every user to the test set and the remaining ones to the training set
    """
    sequences = data.sort_values(by=["user", "initepoch"]).groupby("user")["id"]
    last_sequence = sequences.last()
    train = data[~data.id.isin(last_sequence.values)].copy()
    test = data[data.id.isin(last_sequence.values)].copy()
    return train, test


def f_measure(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def precision(ground_truth, prediction):
    """
    Compute Precision metric
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    precision_score = count_a_in_b_unique(prediction, ground_truth) / float(len(prediction)) if float(len(prediction)) > 0 else 0
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ground_truth, prediction):
    """
    Compute Recall metric
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    recall_score = 0 if len(prediction) == 0 else count_a_in_b_unique(prediction, ground_truth) /    float(
        len(ground_truth)) if float(len(ground_truth)) > 0 else 0
    assert 0 <= recall_score <= 1
    return recall_score


def mrr(ground_truth, prediction):
    """
    Compute Mean Reciprocal Rank metric. Reciprocal Rank is set 0 if no predicted item is in contained the ground truth.
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    rr = 0.
    for rank, p in enumerate(prediction):
        if p in ground_truth:
            rr = 1. / (rank + 1)
            break
    return rr


def count_a_in_b_unique(a, b):
    """
    :param a: list of lists
    :param b: list of lists
    :return: number of elements of a in b
    """
    count = 0
    for el in a:
        if el in b:
            count += 1
    return count


def remove_duplicates(l):
    return [list(x) for x in set(tuple(x) for x in l)]


def sequential_evaluation(recommender,
                          test_sequences,
                          evaluation_functions,
                          users=None,
                          given_k=1,
                          look_ahead=1,
                          top_n=10,
                          scroll=True,
                          step=1):
    """
    Runs sequential evaluation of a Recommender over a set of test sequences
    :param recommender: the instance of the Recommender to test
    :param test_sequences: the set of test sequences
    :param evaluation_functions: list of evaluation metric functions
    :param users: (optional) the list of user ids associated to each test sequence. Required by personalized models like FPMC.
    :param given_k: (optional) the initial size of each user profile, starting from the first interaction in the sequence.
                    If <0, start counting from the end of the sequence. It must be != 0.
    :param look_ahead: (optional) number of subsequent interactions in the sequence to be considered as ground truth.
                    It can be any positive number or 'all' to extend the ground truth until the end of the sequence.
    :param top_n: (optional) size of the recommendation list
    :param scroll: (optional) whether to scroll the ground truth until the end of the sequence.
                If True, expand the user profile and move the ground truth forward of `step` interactions. Recompute and evaluate recommendations every time.
                If False, evaluate recommendations once per sequence without expanding the user profile.
    :param step: (optional) number of interactions that will be added to the user profile at each step of the sequential evaluation.
    :return: the list of the average values for each evaluation metric
    """
    if given_k == 0:
        raise ValueError('given_k must be != 0')
    results = []
    metrics = np.zeros(len(evaluation_functions))
    for i, test_seq in enumerate(test_sequences):
        if users is not None:
            user = users[i]
        else:
            user = None
        logging.debug("Sequence:{}".format(test_seq) )
        if scroll:
            r = sequence_sequential_evaluation(recommender,
                                               test_seq,
                                               evaluation_functions,
                                               user,
                                               given_k,
                                               look_ahead,
                                               top_n,
                                               step)
            metrics += r
            results.append(r)
        else:
            r = evaluate_sequence(recommender,
                                  test_seq,
                                  evaluation_functions,
                                  user,
                                  given_k,
                                  look_ahead,
                                  top_n)
            metrics += r
            results.append(r)
    return metrics / len(test_sequences), np.array(results)


def evaluate_sequence(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n):
    """
    :param recommender: which Recommender to use
    :param seq: the user_profile/ context
    :param given_k: last element used as ground truth. NB if <0 it is interpreted as first elements to keep
    :param evaluation_functions: which function to use to evaluate the rec performance
    :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
    :return: performance of Recommender
    """
    # safety checks

    if given_k < 0:
        given_k = len(seq) + given_k

    user_profile = seq[:given_k]
    logging.debug("-User profile: {}".format(user_profile))

    ground_truth = seq[given_k:]
    # restrict ground truth to look_ahead
    ground_truth = ground_truth[:look_ahead] if look_ahead != 'all' else ground_truth
    ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format
    logging.debug("-Ground truth: {}".format(ground_truth))

    if not user_profile or not ground_truth:
        # if any of the two missing all evaluation functions are 0
        return np.zeros(len(evaluation_functions))

    r = recommender.recommend(user_profile, user)[:top_n]
    logging.debug("-Prediction: {}".format(r))

    if not r:
        # no recommendation found
        return np.zeros(len(evaluation_functions))
    reco_list = recommender.get_recommendation_list(r)

    tmp_results = []
    for f in evaluation_functions:
        tmp_results.append(f(ground_truth, reco_list))


    logging.debug("-".center(50,"-"))

    return np.array(tmp_results)


def sequence_sequential_evaluation(recommender, seq, evaluation_functions, user, given_k, look_ahead, top_n, step):
    if given_k < 0:
        given_k = len(seq) + given_k

    eval_res = 0.0
    eval_cnt = 0
    for gk in range(given_k, len(seq), step):
        eval_res += evaluate_sequence(recommender, seq, evaluation_functions, user, gk, look_ahead, top_n)
        eval_cnt += 1
    precision_metrics.append((eval_res / eval_cnt)[0])
    recall_metrics.append((eval_res / eval_cnt)[1])
    return eval_res / eval_cnt

def getTrueRating(df, user, item):
  try:
    userRows=df[df["user"]==user]
    return userRows.loc[userRows["service"] == item,"value"].values[0]
  except TypeError:
    print(item)
    return 0
  except IndexError:
    print(item)
    return 0
def evaluator(model,df,k):
    items=np.array(['LG400A-100648 (RECC06) TX', 'LG1000 - 100672 IDK', 'LG600B6-100636-IDK', '110000556 DEC',
                    'DM2 Test', 'LG600-B8-110000578 LINAMAR', 'ESTARTA315FV-100626 (E-315', 'Wheelhead1Power', 'Wheelhead2Power',
                    'Wheelhead2Speed_actual', 'X1AxisForce', 'Z1AxisForce', 'CAxisSpeed_actual', 'X1AxisPosition_actual', 'CAxisPower',
                    'X1AxisSpeed_commanded', 'Z1AxisSpeed_commanded', 'grinding-center', 'common-moving-average', 'common-negate', 'common-fft',
                    'grinding-power-stats', 'grinding-specific-energy', 'grinding-force-stats', 'grinding-defect-spark', 'grinding-stiffness',
                    'grinding-material-fed', 'basic-basic-stats', 'basic-quotient', 'grinding-contact-point', 'TDOMONWH2X1Z1', 'TDOMONWH2X1', 'TDOMONWH1X1', 'TDOMONWH2Z1'])
    l_p=[]
    l_r=[]
    l_mrr=[]

    for uuid in df["user"].unique():
        ratings_est = []
        ratings_ui = []
        for i in items:
            d=getTrueRating(df, uuid, i)
            p=model.predict(uuid, i,r_ui=getTrueRating(df,uuid,i))
            ratings_est.append(p.est)
            ratings_ui.append(p.r_ui)
        ratings_est=np.array(ratings_est)
        ratings_ui = np.array(ratings_ui)
        recommendation=items[ratings_est.argsort()[::-1]][:k]
        actual_top=items[ratings_ui.argsort()[::-1]][:k]
        print("-----------------------------------------------")
        print("user:", uuid)
        print("Recommendation", recommendation, "ActualTop", actual_top)
        print("Ratings Estimated", list(ratings_est[ratings_est.argsort()[::-1]])[:k])
        print("Ratingsui", list(ratings_ui[ratings_ui.argsort()[::-1]])[:k])
        p, r, m = precision(actual_top, recommendation), recall(actual_top, recommendation), mrr(actual_top,
                                                                                                 recommendation)

        print("Precision", p, "Recall", r, "MRR", m)
        print("-------------------------------------------------")
        l_p.append(p);l_r.append(r);l_mrr.append(m)
    return np.array(l_p).mean(),np.array(l_r).mean(),np.array(l_mrr).mean()


def evaluator_testset(model,testset,df,k, mean_rating,seed):
    items = np.array(
        ['LG400A-100648 (RECC06) TX', 'LG1000 - 100672 IDK', 'LG600B6-100636-IDK', '110000556 DEC', 'DM2 Test',
         'LG600-B8-110000578 LINAMAR', 'ESTARTA315FV-100626 (E-315', 'Wheelhead1Power', 'Wheelhead2Power',
         'Wheelhead2Speed_actual', 'X1AxisForce', 'Z1AxisForce', 'CAxisSpeed_actual', 'X1AxisPosition_actual',
         'CAxisPower', 'X1AxisSpeed_commanded', 'Z1AxisSpeed_commanded', 'grinding-center', 'common-moving-average',
         'common-negate', 'common-fft', 'grinding-power-stats', 'grinding-specific-energy', 'grinding-force-stats',
         'grinding-defect-spark', 'grinding-stiffness', 'grinding-material-fed', 'basic-basic-stats', 'basic-quotient',
         'grinding-contact-point', 'TDOMONWH2X1Z1', 'TDOMONWH2X1', 'TDOMONWH1X1', 'TDOMONWH2Z1'])

    l_p = []
    l_r = []
    l_mrr = []
    usersTest = [x[0] for x in testset]
    #commented for test without context
    #itemTest = np.unique( np.array([x[1] for x in testset]))

    np.random.seed(seed)
    itemTest = np.random.choice(items, size=10, replace=False)

    print(usersTest,itemTest)
    #mean_rating=0.01

    for uuid in set(usersTest):
        ratings_est = []
        ratings_ui = []
        for i in itemTest:
            #d = getTrueRating(df, uuid, i)
            p = model.predict(uuid, i, r_ui=getTrueRating(df, uuid, i))
            ratings_est.append(p.est)
            ratings_ui.append(p.r_ui)
        #Filtro solo los que sean mayor al rating promedio

        ratings_est = np.array(ratings_est)
        #itemTest1 = itemTest[ratings_est > mean_rating]
        #ratings_est=ratings_est[ratings_est>mean_rating]
        itemTest1 = itemTest
        ratings_est=ratings_est

        ratings_ui = np.array(ratings_ui)
        itemTest2 = itemTest[ratings_ui > mean_rating]
        ratings_ui=ratings_ui[ratings_ui > mean_rating]
        #itemTest2 = itemTest
        #ratings_ui = ratings_ui

        recommendation = itemTest1[ratings_est.argsort()[::-1]][:k]
        actual_top = itemTest2[ratings_ui.argsort()[::-1]][:k]
        print("-----------------------------------------------")
        print("user:", uuid, "k", k)
        print("Recommendation", recommendation, "ActualTop", actual_top)
        print("Ratings Estimated", list(ratings_est[ratings_est.argsort()[::-1]])[:k])
        print("Ratingsui", list(ratings_ui[ratings_ui.argsort()[::-1]])[:k])
        p, r, m = precision(actual_top, recommendation), recall(actual_top, recommendation), mrr(actual_top,recommendation)

        print("Precision",p,"Recall",r,"MRR",m)
        print("-------------------------------------------------")
        l_p.append(p);
        l_r.append(r);
        l_mrr.append(m)
    return np.array(l_p).mean(), np.array(l_r).mean(), np.array(l_mrr).mean()

#old that I used for context
def evaluator_testset_context(model,testset,df,k, mean_rating,seed):
    items = np.array(
        ['LG400A-100648 (RECC06) TX', 'LG1000 - 100672 IDK', 'LG600B6-100636-IDK', '110000556 DEC', 'DM2 Test',
         'LG600-B8-110000578 LINAMAR', 'ESTARTA315FV-100626 (E-315', 'Wheelhead1Power', 'Wheelhead2Power',
         'Wheelhead2Speed_actual', 'X1AxisForce', 'Z1AxisForce', 'CAxisSpeed_actual', 'X1AxisPosition_actual',
         'CAxisPower', 'X1AxisSpeed_commanded', 'Z1AxisSpeed_commanded', 'grinding-center', 'common-moving-average',
         'common-negate', 'common-fft', 'grinding-power-stats', 'grinding-specific-energy', 'grinding-force-stats',
         'grinding-defect-spark', 'grinding-stiffness', 'grinding-material-fed', 'basic-basic-stats', 'basic-quotient',
         'grinding-contact-point', 'TDOMONWH2X1Z1', 'TDOMONWH2X1', 'TDOMONWH1X1', 'TDOMONWH2Z1'])

    l_p = []
    l_r = []
    l_mrr = []
    usersTest = [x[0] for x in testset]
    itemTest = np.unique( np.array([x[1] for x in testset]))

    #commented for test with context
    #np.random.seed(seed)
    #itemTest = np.random.choice(items, size=10, replace=False)

    print(usersTest,itemTest)
    #mean_rating=0.01

    for uuid in set(usersTest):
        ratings_est = []
        ratings_ui = []
        for i in itemTest:
            #d = getTrueRating(df, uuid, i)
            p = model.predict(uuid, i, r_ui=getTrueRating(df, uuid, i))
            ratings_est.append(p.est)
            ratings_ui.append(p.r_ui)
        #Filtro solo los que sean mayor al rating promedio

        ratings_est = np.array(ratings_est)
        #itemTest1 = itemTest[ratings_est > mean_rating]
        #ratings_est=ratings_est[ratings_est>mean_rating]
        itemTest1 = itemTest
        ratings_est=ratings_est

        ratings_ui = np.array(ratings_ui)
        itemTest2 = itemTest[ratings_ui > mean_rating]
        ratings_ui=ratings_ui[ratings_ui > mean_rating]
        #itemTest2 = itemTest
        #ratings_ui = ratings_ui

        recommendation = itemTest1[ratings_est.argsort()[::-1]][:k]
        actual_top = itemTest2[ratings_ui.argsort()[::-1]][:k]
        print("-----------------------------------------------")
        print("user:", uuid, "k", k)
        print("Recommendation", recommendation, "ActualTop", actual_top)
        print("Ratings Estimated", list(ratings_est[ratings_est.argsort()[::-1]])[:k])
        print("Ratingsui", list(ratings_ui[ratings_ui.argsort()[::-1]])[:k])
        p, r, m = precision(actual_top, recommendation), recall(actual_top, recommendation), mrr(actual_top,recommendation)

        print("Precision",p,"Recall",r,"MRR",m)
        print("-------------------------------------------------")
        l_p.append(p);
        l_r.append(r);
        l_mrr.append(m)
    return np.array(l_p).mean(), np.array(l_r).mean(), np.array(l_mrr).mean()


def predict(model,df,k, uuid):
   # items = np.array(['arabiar tea', 'capuccino', 'capuccino decaffeinated', 'chocolate', 'chocolate_milk', 'cortado',
   #                   'cortado decaffeinated', 'decaffeinated', 'expresso', 'hazelnut_cappuccino', 'light_coffee',
   #                   'long_coffee', 'long_decaffeinated', 'milk_coffee', 'milk_coffee kafeinagabe', 'txokolate white'])
    items = np.array(
       ['LG400A-100648 (RECC06) TX', 'LG1000 - 100672 IDK', 'LG600B6-100636-IDK', '110000556 DEC', 'DM2 Test',
        'LG600-B8-110000578 LINAMAR', 'ESTARTA315FV-100626 (E-315', 'Wheelhead1Power', 'Wheelhead2Power',
        'Wheelhead2Speed_actual', 'X1AxisForce', 'Z1AxisForce', 'CAxisSpeed_actual', 'X1AxisPosition_actual',
        'CAxisPower', 'X1AxisSpeed_commanded', 'Z1AxisSpeed_commanded', 'grinding-center', 'common-moving-average',
        'common-negate', 'common-fft', 'grinding-power-stats', 'grinding-specific-energy', 'grinding-force-stats',
        'grinding-defect-spark', 'grinding-stiffness', 'grinding-material-fed', 'basic-basic-stats', 'basic-quotient',
        'grinding-contact-point', 'TDOMONWH2X1Z1', 'TDOMONWH2X1', 'TDOMONWH1X1', 'TDOMONWH2Z1'])

    ratings_est = []
    ratings_ui = []
    for i in items:
        p = model.predict(uuid, i, r_ui=getTrueRating(df, uuid, i))
        ratings_est.append(p.est)
        ratings_ui.append(p.r_ui)
    ratings_est = np.array(ratings_est)
    ratings_ui = np.array(ratings_ui)
    recommendation = items[ratings_est.argsort()[::-1]][:k]
    actual_top = items[ratings_ui.argsort()[::-1]][:k]
    print(recommendation,ratings_est[ratings_est.argsort()[::-1]])
    print(actual_top, ratings_ui[ratings_ui.argsort()[::-1]])

def precision_recall_at_k(predictions, k=3, threshold=0.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    print(user_est_true)
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value in descending order.
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Get the top-K predictions.
        top_k_estimates = [x[0] for x in user_ratings[:k]]
        # Get the top-K true ratings.
        top_k_true_ratings = [x[1] for x in user_ratings[:k]]
        # Calculate the number of relevant items at K.
        num_relevant_items = sum([1 if true_r >= threshold else 0 for true_r in top_k_true_ratings])
        # Calculate the number of recommended items that are relevant.
        num_recommended_relevant_items = sum([1 if est >= threshold else 0 for est in top_k_estimates])
        # Calculate the number of relevant items that were recommended.
        num_recommended_items = len(top_k_estimates)
        # Calculate precision and recall at K.
        precisions[uid] = num_recommended_relevant_items / num_recommended_items if num_recommended_items > 0 else 0
        recalls[uid] = num_recommended_relevant_items / num_relevant_items if num_relevant_items > 0 else 0

    # Compute the average precision and recall at K across all users.
    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)

    return precision, recall