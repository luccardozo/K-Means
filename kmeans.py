import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd


def cat_to_num(d, categorical):
    for cat in categorical:
        d[cat] = pd.factorize(d[cat])[0]
    return d


def init_centroids(x, k):
    centroids = pd.DataFrame()
    for i in range(0, k):
        while len(centroids.drop_duplicates()) != i + 1:
            rand = rd.randint(0, x.shape[0])
            centroids = centroids.append(x.iloc[rand])

    return centroids


def kmeans(d, k):
    """
    :param d: Dataset
    :param k:  K-Clusters
    :param dtype:  Data type - 'string' for categorical features
                               'numeric' for numerical features

    :return: Clusters
    """
    # Escolhe centroides aleatorios
    centroids = init_centroids(d, k)
    m = d.shape[0]

    # compute euclidian distances and assign clusters
    for n in range(5):  # While(not alteration):
        #
        # Implementar iterações continuas até que não
        #   haja alteração nos clusters
        #
        distance = pd.DataFrame()
        for i in range(k):
            tempDist = np.sum((d - centroids.iloc[i]) ** 2, axis=1)
            distance = pd.concat([distance, tempDist], axis=1, ignore_index=True)

        min_dist = distance.idxmin(axis='columns')

        # Separa os clusters e adiciona as linhas no dicionario
        clusters = {}
        for i in range(k):
            clusters[i] = pd.DataFrame()
        for i in range(m):
            clusters[min_dist.iloc[i]] = clusters[min_dist.iloc[i]].append(d.iloc[i])

        centroids = centroids[0:0]  # Apaga os centroides antigos
        # Ajusta os centroides
        for i in range(k):
            centroids = centroids.append(clusters[i].mean(), ignore_index=True)

    return clusters, WCSS(k, clusters, centroids)


def WCSS(k, clusters, centroids):
    wcss = 0
    for i in range(k):
        wcss += np.sum((clusters[i] - centroids.iloc[i]) ** 2)
    return wcss.mean()


dataset1 = cat_to_num(pd.read_csv('data\SocioDemographic_Vars.txt', sep='\t'), ["Gender", "Only.child", "Education"])
dataset = pd.read_csv('data\Phobias_Vars.txt', sep='\t')

print(kmeans(dataset, 4))

""""
EVITANDO PERDER OS TRABALHOS

if dtype == 'string':
    tempDist = np.sum(d.eq(centroids.iloc[i], axis=1) * -1, axis=1)

if dtype == 'string':
    centroids = centroids.append(clusters[i].mode(axis=0).iloc[0], ignore_index=True)
    
    
"""
