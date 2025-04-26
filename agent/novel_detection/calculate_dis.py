import pickle
import os
import sys
# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np

import os

from tqdm import tqdm

import json



import scanpy as sc

from collections import Counter

from scipy.spatial.distance import euclidean

from scAgent.memory.search_db import query

from config import MODEL_PATHS, DICT_PATHS


# 计算欧氏距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# 计算欧氏距离的平方
def squared_euclidean_distance(a, b):
    return np.sum((a - b) ** 2)


def search_nearest(organcellembs, emb):
    # 存储距离和对应的category
    distances = []
    for category, emblist in organcellembs.items():
        for vec in emblist:
            dist = euclidean(emb, vec)
            distances.append((dist, category))

    return distances


def statistic_dis_lora(organ, input_emblist):
    # with open(f'/data/lpg/codes/scAgent/share/agent/openClassifier/tab_database/{organ}/train.pkl', 'rb') as f:
    #     organcellembs = pickle.load(f)

    # emblist_mean = {}

    # # 计算每个类别的均值向量
    # for category, emblist in organcellembs.items():
    #     emblist_mean[category] = np.mean(emblist, axis=0)


    with open(DICT_PATHS["center_lora"], 'rb') as f:
        organcellembs = pickle.load(f)

    emblist_mean=organcellembs[organ]


    distanceslist = []
    # 循环输入的每个向量
    for emb in tqdm(input_emblist, desc=f"Processing", unit="emb"):
        mean_distances = []
        for category, emb_mean in emblist_mean.items():
            # mean_distance = euclidean_distance(emb, emb_mean)
            mean_distance = squared_euclidean_distance(emb, emb_mean)
            mean_distances.append((category, mean_distance))

        # 对 distances 按照距离从小到大排序
        cluster_center_distances = sorted(mean_distances, key=lambda x: x[1])


        top10_nearest_distances = query([emb], search_num=10, is_lora=True, search_type="cells", search_data="sctab", tissue=organ)[0]

        # print(top10_nearest_distances)
        # print(result)

        distanceslist.append([cluster_center_distances, top10_nearest_distances])

    return distanceslist


def statistic_dis_nolora(organ, input_emblist):
    # with open(f'/data/lpg/codes/scAgent/share/agent/openClassifier/tab_database/{organ}/train_nolora.pkl', 'rb') as f:
    #     organcellembs = pickle.load(f)

    # emblist_mean = {}

    # for category, emblist in organcellembs.items():
    #     emblist_mean[category] = np.mean(emblist, axis=0)


    with open(DICT_PATHS["center_nolora"], 'rb') as f:
        organcellembs = pickle.load(f)

    emblist_mean=organcellembs[organ]

    distanceslist = []
    for emb in tqdm(input_emblist, desc=f"Processing", unit="emb"):
        mean_distances = []
        for category, emb_mean in emblist_mean.items():
            # mean_distance = euclidean_distance(emb, emb_mean)
            mean_distance = squared_euclidean_distance(emb, emb_mean)
            mean_distances.append((category, mean_distance))

        # 对 distances 按照距离从小到大排序
        cluster_center_distances = sorted(mean_distances, key=lambda x: x[1])

        cluster_center_nearest = cluster_center_distances[0]


        top10_nearest_distances = \
        query([emb], search_num=10, is_lora=False, search_type="cells", search_data="sctab", tissue=organ)[0]

        # print(top10_nearest_distances)
        # print(result)

        # 计算最近的十个向量的平均距离
        average_distance = sum(dist for dist, _ in top10_nearest_distances) / len(top10_nearest_distances)

        # 找出其中最多的细胞类型
        categories = [category for _, category in top10_nearest_distances]
        most_common_category = Counter(categories).most_common(1)[0][0]

        distanceslist.append([cluster_center_nearest, most_common_category, average_distance])

    return distanceslist