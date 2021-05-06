import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd


def init_centroids(x, k):
    centroids = pd.DataFrame()
    for i in range(0, k):
        while len(centroids.drop_duplicates()) != i + 1:
            rand = rd.randint(0, x.shape[0])
            centroids = centroids.append(x.iloc[rand])

    return centroids


def kmeans(d, k):
    """
    :param d: Data
    :param k: K-Clusters
    :return: Clusters
    """
    # Escolhe centroides aleatorios
    centroids = init_centroids(d, k)
    m = d.shape[0]

    # compute euclidian distances and assign clusters
    for n in range(10):  # While(not alteration):
        #
        # Implementar iterações continuas até que não
        #   haja alteração nos clusters
        #
        distance = pd.DataFrame()
        for i in range(k):
            tempDist = np.sum((d - centroids.iloc[i]) ** 2, axis=1)
            distance = pd.concat([distance, tempDist], axis=1, ignore_index=True)

        min_dist = distance.idxmin(axis='columns')
        # adjust the centroids
        clusters = {}
        for i in range(k):
            clusters[i] = pd.DataFrame()
        for i in range(m):
            clusters[min_dist.iloc[i]] = clusters[min_dist.iloc[i]].append(d.iloc[i])

        centroids = centroids[0:0]              # Apaga os centroides antigos
        for i in range(k):
            centroids = centroids.append(clusters[i].mean(), ignore_index=True)

    return clusters


def WCSS(k, y, centroids):
    wcss = 0
    for i in range(k):
        wcss += np.sum((y[i + 1] - centroids[:, i]) ** 2)
    return wcss


dataset = pd.read_csv('data\Phobias_Vars.txt', sep='\t', dtype='Int64')
kmeans(dataset, 4)
