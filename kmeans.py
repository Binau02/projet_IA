import sys
import math
import json


def string_to_array_2d(test):
  test = test.replace(" ", '')
  test = test.replace("[", '')
  test = test[:-2]
  test = test.split(sep = "],")
  for i in range(len(test)):
    test[i] = test[i].split(sep = ",")
    test[i][0] = float(test[i][0])
    test[i][1] = float(test[i][1])
  
  return test

def appartient_cluster(lat, lon, clusters):
  lat = float(lat)
  lon = float(lon)
  clusters = string_to_array_2d(clusters)

  dist_table = []
  cluster_index = 0
  for j in range(len(clusters)): # 0 à 4
    dist = [math.sqrt((clusters[j][0]-lat)**2+(clusters[j][1]-lon)**2),cluster_index] # calculer la distance entre le point et le cluster j
    cluster_index += 1
    dist_table.append(dist)

  min_dist = min(dist_table)
  index = min_dist[1]
  # return index

  # créer un fichier json qui contient les coordonnées du cluster auquel appartient le point
  aDict = {"lat_centroides":clusters[index][0], "lon_centroides":clusters[index][1]}
  jsonString = json.dumps(aDict, indent = 4)
  # jsonFile = open("cluster.json", "w")
  # jsonFile.write(jsonString)
  # jsonFile.close()
  return jsonString

cluster = appartient_cluster(sys.argv[1], sys.argv[2], sys.argv[3])
print(cluster)
