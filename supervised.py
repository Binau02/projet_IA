import pandas as pd
import math
import collections
import matplotlib.pyplot as plt
import time
import json




def knn(csv_file, distance_type = "euclidean", p = None, k = 5, data = [], latitude = None, longitude = None, age = None, weight = None, hour = None, athmo = None, surf = None, lum = None, agglo = None):
  df = pd.read_csv(csv_file, sep = ";")
  df.drop(columns = ['Num_Acc', 'num_veh', 'id_usa', 'date', 'ville', 'id_code_insee', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'description_intersection', 'descr_dispo_secu', 'descr_grav', 'descr_motif_traj', 'descr_type_col', 'an_nais', 'place', 'dept', 'region', 'CODE_REG', 'weeks', 'month', 'days', 'intersection_num', 'motif_num', 'collision_num'], inplace = True)

  labels = df.loc[:, 'gravity'].to_frame()
  df.drop(columns = ['gravity'], inplace = True)

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

