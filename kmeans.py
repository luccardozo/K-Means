import numpy as np
import pandas as pd
import random as rd


def cat_to_num(d, categorical):
    for cat in categorical:
        d[cat] = pd.factorize(d[cat])[0]
    return d


def init_centroids(x, k):
    centroids = pd.DataFrame()
    for i in range(0, k):
        # Garante que não vamos adicionar dois centroides no mesmo lugar inicial. Gerando sempre um novo aleatório.
        while len(centroids.drop_duplicates()) != i + 1:
            rand = rd.randint(0, x.shape[0])  # valor aleatorio entre zero e o numero de linhas (valores de entrada)
            centroids = centroids.append(x.iloc[rand])

    return centroids


def kmeans(d, k):
    """
    :param d: Dataset
    :param k:  K-Clusters

    :return: Clusters
    """
    max_iterations = 100
    decimal_precision = 3
    # Escolhe centroides aleatorios não iguais
    global clusters
    centroids = init_centroids(d, k)

    centroids.astype(float).round(decimals=decimal_precision)
    m = d.shape[0]  # numero de linhas
    changed = True
    curr_iteration = 0
    # Calcula distancia Euclideanea
    while (max_iterations > curr_iteration) and changed:
        curr_iteration += 1
        distance = pd.DataFrame()
        for i in range(k):
            # Serie com a distancia calculada para cada um dos pontos.
            tempDist = np.sum((d - centroids.iloc[i]) ** 2, axis=1)

            # Matriz com a distancia de cada centroide para cada ponto.
            distance = pd.concat([distance, tempDist], axis=1, ignore_index=True)

        min_dist = distance.idxmin(axis='columns')

        # Separa os clusters e adiciona as linhas no dicionario
        clusters = {}
        for i in range(k):
            clusters[i] = pd.DataFrame()
        for i in range(m):
            clusters[min_dist.iloc[i]] = clusters[min_dist.iloc[i]].append(d.iloc[i])

        new_centroid = pd.DataFrame()
        new_centroid = new_centroid[0:0]  # Apaga os centroides antigos
        for i in range(k):
            new_centroid = new_centroid.append(clusters[i].mean(), ignore_index=True)

        new_centroid = new_centroid.astype(float).round(decimals=decimal_precision)

        if centroids.equals(new_centroid):
            changed = False
        else:
            centroids = new_centroid

    return clusters, curr_iteration, WCSS(k, clusters, centroids)


def WCSS(k, c, centroids):
    wcss = 0
    for j in range(k):
        wcss += np.sum((c[j] - centroids.iloc[j]) ** 2)
    return wcss.mean()


""""
EVITANDO PERDER OS TRABALHOS

if dtype == 'string':
    tempDist = np.sum(d.eq(centroids.iloc[i], axis=1) * -1, axis=1)

if dtype == 'string':
     new_centroid = new_centroid.append(clusters[i].mode(axis=0).iloc[0], ignore_index=True)
    
"""
