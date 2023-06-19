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


def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians

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
    return index

    # créer un fichier json qui contient les coordonnées du cluster auquel appartient le point
    # aDict = {"lat_centroides":clusters[index][0], "lon_centroides":clusters[index][1]}
    # jsonString = json.dumps(aDict)
    # jsonFile = open("cluster.json", "w")
    # jsonFile.write(jsonString)
    # jsonFile.close()


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

        for i in range(len(latitude)):
            dist_table = []
            cluster_index = 0
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
                        # if unit == 'miles':
                        dist = [round(distance, 2),cluster_index]
                        
                        # if unit == 'kilometers':
                            # return round(distance * 1.609344, 2)
                # ecrit la distance de manahttan entre le point et le cluster j dans la table dist_table
                # dist = [abs((clusters[j][0]-latitude[i])+(clusters[j][1]-longitude[i]))**(1/mink),cluster_index]
                # manatthan
                # dist = [abs((clusters[j][0]-latitude[i])+(clusters[j][1]-longitude[i])),cluster_index]
                # euclidienne
                # dist = [math.sqrt((clusters[j][0]-latitude[i])**2+(clusters[j][1]-longitude[i])**2),cluster_index]
                # print(dist)
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
    
    return clusters


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


df = pd.read_csv('data/stat_acc_V3.csv', sep =";")  
print("data loaded !")  
df = fit_departement(df)
print("data fitted !")
st = time.time()
clusters = KMEANS(5,df, 10,methode=1)
et = time.time()
elapsed_time = et - st
print('Execution time manual:', elapsed_time, 'seconds')
# print(clusters)
# print("KMEANS ok !")
# print("final clusters : ", clusters)
print("plotting clusters ...")
plot_clusters(df,clusters,'export/manuel.png')
print("plotting done !")
plt.close()
st = time.time()
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(df[["latitude","longitude"]])
et = time.time()
elapsed_time = et - st
print('Execution time sklearn:', elapsed_time, 'seconds')

# print(kmeans.cluster_centers_)
plot_clusters(df,kmeans.cluster_centers_,'export/sklearn.png')
plt.close()

