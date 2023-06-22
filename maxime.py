# import libraries
import pandas as pd
import random
import math
from statistics import mean 
import json
import plotly.express as px
import matplotlib.pyplot as plt
from numpy import sin, cos, arccos, pi, round
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians

# script sur l'apprentissage non supervisé de la fonctionnalité 4
def appartient_cluster(lat,lon, clusters):
    dist_table = []
    cluster_index = 0
    for j in range(len(clusters)): # 0 à k
        dist = [math.sqrt((clusters[j][0]-lat)**2+(clusters[j][1]-lon)**2),cluster_index] # calculer la distance entre le point et le cluster j
        cluster_index += 1
        dist_table.append(dist)

    min_dist = min(dist_table)
    index = min_dist[1]
    # return index

    # créer un fichier json qui contient les coordonnées du cluster auquel appartient le point
    aDict = {"lat_centroides":clusters[index][0], "lon_centroides":clusters[index][1]}
    jsonString = json.dumps(aDict)
    jsonFile = open("cluster.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def KMEANS(k,df, nb_iteration = 10,methode=3,mink=2):

    # get the first centroids coordinates
    clusters = [] # contient les coordonnées des clusters
    latitude = df["latitude"]
    longitude = df["longitude"]
    for i in range(k):
        random_index = random.randrange(len(latitude))
        clusters.append([latitude[random_index],longitude[random_index]])

    # get the distance between each point and each centroid

    for p in range(nb_iteration):
        clusters_you_are_in = []
        i = 0
        for i in range(len(latitude)):
            dist_table = []
            cluster_index = 0
            j = 0
            for j in range(k):
                match methode:
                    case 1:
                        dist = [abs((clusters[j][0]-latitude[i])+(clusters[j][1]-longitude[i]))**(1/mink),cluster_index]
                    case 2:
                        dist = [abs((clusters[j][0]-latitude[i])+(clusters[j][1]-longitude[i])),cluster_index]
                    case 3:
                        dist = [math.sqrt((clusters[j][0]-latitude[i])**2+(clusters[j][1]-longitude[i])**2),cluster_index]
                    case 4: # distance de haversine
                        theta = clusters[j][1] - longitude[i]
    
                        distance = 60 * 1.1515 * rad2deg(
                            arccos(
                                (sin(deg2rad(latitude[i])) * sin(deg2rad(clusters[j][0]))) + 
                                (cos(deg2rad(latitude[i])) * cos(deg2rad(clusters[j][0])) * cos(deg2rad(theta)))
                            )
                        )
                        dist = [round(distance, 2),cluster_index]

                cluster_index += 1
                dist_table.append(dist)

            min_dist = min(dist_table)
            clusters_you_are_in.append(min_dist[1])
        table_lat = []
        table_lon = []

        for m in range(k):  # 0 1 2 3 4
            for n in range(len(clusters_you_are_in)): # 0 à 73640
                if(clusters_you_are_in[n] == m): # recuperer les coordonnées des points qui sont dans le cluster m
                    table_lat.append(latitude[n])
                    table_lon.append(longitude[n])
            new_coord_lat = mean(table_lat) # calculer la moyenne des coordonnées des points qui sont dans le cluster m
            new_coord_lon = mean(table_lon)
            clusters[m][0] = new_coord_lat # remplacer les coordonnées du cluster m par la moyenne des coordonnées des points qui sont dans le cluster m
            clusters[m][1] = new_coord_lon
            table_lat = []
            table_lon = []
        # print("Coordonnées des clusters à l'iteration", p, " : " ,clusters)
    test = [clusters,clusters_you_are_in]
    return test

def fit_departement(df):
    for i in range(len(df["latitude"])):
        if(df["latitude"][i] < 40 ):
            df.drop(i, inplace=True,axis=0)
        else:
            if(df["longitude"][i] < -10 or df["longitude"][i]> 20):
                df.drop(i, inplace=True,axis=0)

    df.reset_index(drop=True, inplace=True)
    return df

def plot_clusters(df,clusters,path):
    palette = ['bo','ro','go','co','mo','yo','ko','wo']
    for i in range(len(df["longitude"])):
        index = appartient_cluster(df["latitude"][i],df["longitude"][i],clusters)
        match index:
            case 0:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[0],markersize=0.1)
            case 1:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[1],markersize=0.1)
            case 2:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[2],markersize=0.1)
            case 3:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[3],markersize=0.1)
            case 4:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[4],markersize=0.1)
            case 5:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[5],markersize=0.1)
            case 6:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[6],markersize=0.1)
            case 7:
                plt.plot(df["longitude"][i], df["latitude"][i], palette[7],markersize=0.1)
            
    for i in range(len(clusters)):
        plt.plot(clusters[i][1], clusters[i][0], palette[i],markersize=10)

    plt.ylabel('latitude et longitude')
    plt.savefig(path)

def metrics_test(df):

    path_silouhette1 = 'export/silouhette_sklearn.png'
    path_silouhette2 = 'export/silouhette_scratch.png'
    path_calinski1 = 'export/calinski_sklearn.png'
    path_calinski2 = 'export/calinski_scratch.png'
    path_davies1 = 'export/davies_sklearn.png'
    path_davies2 = 'export/davies_scratch.png'

    for i in range(6):
        kmeans = KMeans(n_clusters=i+2, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
        silouhette = silhouette_score(df[["latitude","longitude"]], kmeans.labels_)
        plt.plot(i+2,silouhette,'-gs')
    plt.ylabel('score silouette')
    plt.savefig(path_silouhette1)
    plt.close()

    for i in range(6):
        result = KMEANS(i+2,df, nb_iteration=10,methode=1)
        labels = result[1]
        silouhette = silhouette_score(df[["latitude","longitude"]], labels)
        plt.plot(i+2,silouhette,'-gs')
    plt.ylabel('score silouette')
    plt.savefig(path_silouhette2)
    plt.close()
    print("silouhette ok")
    for i in range(6):
        kmeans = KMeans(n_clusters=i+2, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
        calinski = calinski_harabasz_score(df[["latitude","longitude"]], kmeans.labels_)
        print("calinski score : ", calinski)
        plt.plot(i+2,calinski,'-gs')
    plt.ylabel('score calinski')
    plt.savefig(path_calinski1)
    plt.close()

    for i in range(6):
        result = KMEANS(i+2,df, nb_iteration=10,methode=1)
        labels = result[1]
        calinski = calinski_harabasz_score(df[["latitude","longitude"]], labels)
        plt.plot(i+2,calinski,'-gs')
    plt.ylabel('score calinski')
    plt.savefig(path_calinski2)
    plt.close()
    print("calinski ok")

    for i in range(6):
        kmeans = KMeans(n_clusters=i+2, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
        davies = davies_bouldin_score(df[["latitude","longitude"]], kmeans.labels_)
        plt.plot(i+2,davies,'-gs')
    plt.ylabel('score davies')
    plt.savefig(path_davies1)
    plt.close()

    for i in range(6):
        result = KMEANS(i+2,df, nb_iteration=10,methode=1)
        labels = result[1]
        davies = davies_bouldin_score(df[["latitude","longitude"]], labels)
        plt.plot(i+2,davies,'-gs')
    plt.ylabel('score davies')
    plt.savefig(path_davies2)
    plt.close()
    print("davies ok")

# fit data
# df = pd.read_csv('data/stat_acc_V3.csv', sep =";")  
# print("data loaded !")  
# df = fit_departement(df)
# print("data fitted !")

# faire les tests de metrics

# metrics_test(df)

# afficher les clusters sur les maps

# tab_paths = ['export/manual_2_clusters.png','export/manual_3_clusters.png','export/manual_4_clusters.png','export/manual_5_clusters.png','export/manual_6_clusters.png']
# tab_paths2 = ['export/sklearn_2_clusters.png','export/sklearn_3_clusters.png','export/sklearn_4_clusters.png','export/sklearn_5_clusters.png','export/sklearn_6_clusters.png']
# for i in range(5):
#     result = KMEANS(i+2,df, nb_iteration=3,methode=1)
#     clusters = result[0]
#     result = KMeans(n_clusters=i+2, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
#     clusters2 = result.cluster_centers_
#     plot_clusters(df,clusters,tab_paths[i])
#     plt.close()
#     plot_clusters(df,clusters2,tab_paths2[i])
#     plt.close()


# comparer les temps d'execution

# st = time.time()
# kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
# et = time.time()
# elapsed_time = et - st
# print('Execution time sklearn:', elapsed_time, 'seconds')
# print(kmeans.cluster_centers_)
# appartient_cluster(df["latitude"][10000],df["longitude"][10000],kmeans.cluster_centers_)

# st = time.time()
# result = KMEANS(5,df, nb_iteration=1,methode=1)
# et = time.time()
# elapsed_time = et - st
# clusters = result[0]
# labels = result[1]
# print(clusters)
# print(labels)
# print('Execution time manual:', elapsed_time, 'seconds')


