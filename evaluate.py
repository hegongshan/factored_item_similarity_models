import heapq
import math

import numpy as np


def evaluate_model(model, testRatings, testNegatives, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs, arhrs = [], [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg, arhr) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        arhrs.append(arhr)
    return hits, ndcgs, arhrs


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    test_item = rating[1]

    items.append(test_item)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict(users, np.array(items))

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    rank_list = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = get_hit_ratio(rank_list, test_item)

    ndcg = get_NDCG(rank_list, test_item)

    arhr = get_ARHR(rank_list, test_item)
    return hr, ndcg, arhr


def get_hit_ratio(rank_list, test_item):
    if test_item in rank_list:
        return 1
    return 0


def get_ARHR(rank_list, test_item):
    for idx, item in enumerate(rank_list):
        if item == test_item:
            return 1 / (idx + 1)
    return 0


def get_NDCG(rank_list, test_item):
    for idx, item in enumerate(rank_list):
        if item == test_item:
            return math.log(2) / math.log(idx + 2)
    return 0
