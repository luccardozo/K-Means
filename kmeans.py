import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd


def init_centroids(x, k):
    centroids = pd.DataFrame()
    for i in range(0, k):
        # Garante que não vamos adicionar dois centroides no mesmo lugar inicial. Gerando sempre um novo aleatório.
        while len(centroids.drop_duplicates()) != i + 1:
            rand = rd.randint(0, x.shape[0]) #valor aleatorio entre zero e o numero de linhas (valores de entrada)
            centroids = centroids.append(x.iloc[rand])

    return centroids


def kmeans(d, k, max_iterations=10, decimal_precision=3):
    """
    :param d: Data
    :param k: K-Clusters
    :return: Clusters
    """
    # Escolhe centroides aleatorios não iguais
    global clusters
    centroids = init_centroids(d, k)
    centroids.astype(float).round(decimals=decimal_precision)
    m = d.shape[0] #numero de linhas
    changed = True
    curr_iteration = 0
    # Calcula distancia Euclideanea
    while ((max_iterations > curr_iteration) and changed):
        curr_iteration += 1
        distance = pd.DataFrame()
        for i in range(k):
            #Serie com a distancia calculada para cada um dos pontos.
            tempDist = np.sum((d - centroids.iloc[i]) ** 2, axis=1)
            #Matriz com a distancia de cada centroide para cada ponto.
            distance = pd.concat([distance, tempDist], axis=1, ignore_index=True)

        #Cada ponte aponta pra o cluster que pertence
        min_dist = distance.idxmin(axis='columns')

        clusters = {}
        #Ajusta os centroides
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

    return clusters, curr_iteration


def WCSS(k, y, centroids):
    wcss = 0
    for i in range(k):
        wcss += np.sum((y[i + 1] - centroids[:, i]) ** 2)
    return wcss


dataset = pd.read_csv('data\Phobias_Vars.txt', sep='\t', dtype='Int64')
kmeans(dataset, 4)
