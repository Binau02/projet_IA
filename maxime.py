import pandas as pd
import random
import math
from statistics import mean 
import json

df = pd.read_csv('data/stat_acc_V3.csv', sep =";")

# script sur l'apprentissage non supervisé
def appartient_cluster(lat,lon, clusters):
    dist_table = []
    cluster_index = 0
    for j in range(len(clusters)): # 0 à 4
        dist = [math.sqrt((clusters[j][0]-lat)**2+(clusters[j][1]-lon)**2),cluster_index] # calculer la distance entre le point et le cluster j
        cluster_index += 1
        dist_table.append(dist)

    min_dist = min(dist_table)
    index = min_dist[1]
    # créer un fichier json qui contient les coordonnées du cluster auquel appartient le point
    aDict = {"lat_centroides":clusters[index][0], "lon_centroides":clusters[index][1]}
    jsonString = json.dumps(aDict)
    jsonFile = open("cluster.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def KMEANS(k,df, nb_iteration = 10):

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

        for i in range(len(latitude)):
            dist_table = []
            cluster_index = 0
            for j in range(k):
                dist = [math.sqrt((clusters[j][0]-latitude[i])**2+(clusters[j][1]-longitude[i])**2),cluster_index]
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
        print("Coordonnées des clusters à l'iteration", p, " : " ,clusters)
    
    return clusters
        

        
clusters = KMEANS(5,df, 2)
print("final clusters : ", clusters)

appartient_cluster(48.8566969,2.3514616,clusters)


