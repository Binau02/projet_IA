import pandas as pd
import math
import collections
import matplotlib.pyplot as plt
import time
import json

from sklearn.neighbors import KNeighborsClassifier




def knn(csv_file, distance_type = "euclidean", p = None, k = 11, data = [], latitude = None, longitude = None, age = None, weight = None, hour = None, athmo = None, surf = None, lum = None, agglo = None):
  if type(data) == "str":
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split(',')

  df = pd.read_csv(csv_file, sep = ";")
  df.drop(columns = ['Num_Acc', 'num_veh', 'id_usa', 'date', 'ville', 'id_code_insee', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'description_intersection', 'descr_dispo_secu', 'descr_grav', 'descr_motif_traj', 'descr_type_col', 'an_nais', 'place', 'dept', 'region', 'CODE_REG', 'weeks', 'month', 'days', 'intersection_num', 'motif_num', 'collision_num'], inplace = True)

  labels = df.loc[:, 'gravity'].to_frame()
  df.drop(columns = ['gravity'], inplace = True)
  df["weight"] /= 1000

  n = len(df)
  distances = []
  for i in range(n):
    row = df.iloc[i]
    d = 0
    for j in range(9):
      if distance_type == "euclidean":
        d += (row[j] - data[j])**2
      elif distance_type == "manhattan":
        d += abs(row[j] - data[j])
      elif distance_type == "minkowski":
        d += abs(row[j] - data[j])
      elif distance_type == "hamming":
        d += int(row[j] != data[j])
      else:
        raise ValueError("The only distance_type accepted are : euclidean, manhattan, minkowski, hamming")
    if (distance_type == "euclidean"):
      d = math.sqrt(d)
    elif distance_type == "minkowski":
      if p == None:
        raise ValueError("If you choose the Minkowski distance_type, please provide a value for p")
      d = d**(1/p)
    distances.append((d, i))

  top_distances = sorted(distances)[:k]
  sum = 0
  results = []
  for i in range(k):
    sum += labels.iloc[top_distances[i][1]][0]
    results.append(labels.iloc[top_distances[i][1]][0])

  n = collections.Counter(results).most_common(1)[0][0]

  return_dict = {"gravity_mean" : str(sum/k), "grivity_best" : str(n)}

  json_object = json.dumps(return_dict, indent = 4) 

  return json_object

st = time.time()
df = pd.read_csv("data/stat_acc_V3.csv", sep = ";")
df.drop(columns = ['Num_Acc', 'num_veh', 'id_usa', 'date', 'ville', 'id_code_insee', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'description_intersection', 'descr_dispo_secu', 'descr_grav', 'descr_motif_traj', 'descr_type_col', 'an_nais', 'place', 'dept', 'region', 'CODE_REG', 'weeks', 'month', 'days', 'intersection_num', 'motif_num', 'collision_num'], inplace = True)
df["weight"] /= 1000

X = df.drop('gravity', axis=1)
y = df['gravity']

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
d = {
  "latitude" : [47.633331],
  "longitude" : [6.86667],
  "age" : [21],
  "weight" : [1],
  "hours" : [14],
  "athmo_num" : [6],
  "etat_surf_num" : [7],
  "lum_num" : [4],
  "agglo_num" : [1]
}
test = pd.DataFrame(data=d)
print(neigh.predict(test))

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

st = time.time()
print(knn("data/stat_acc_V3.csv", distance_type = "euclidean", data = [47.633331, 6.86667, 21, 1, 14, 6, 7, 4, 1]))
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')