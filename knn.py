import sys

import pandas as pd
import math
import collections
import json

from sklearn.neighbors import KNeighborsClassifier



def knn(csv_file, distance_type = "euclidean", p = None, k = 11, data = []):
  if type(data) == type(""):
    data = data.replace(' ', '')
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split(',')

  df = pd.read_csv(csv_file, sep = ";")
  df.drop(columns = ['Num_Acc', 'num_veh', 'id_usa', 'date', 'ville', 'id_code_insee', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'description_intersection', 'descr_dispo_secu', 'descr_grav', 'descr_motif_traj', 'descr_type_col', 'an_nais', 'place', 'dept', 'region', 'CODE_REG', 'weeks', 'month', 'days', 'intersection_num', 'motif_num', 'collision_num'], inplace = True)

  df["weight"] /= 1000
  X = df.drop('gravity', axis=1)
  y = df['gravity']

  neigh = KNeighborsClassifier(n_neighbors=11)
  neigh.fit(X, y)

  d = {
    "latitude" : [data[0]],
    "longitude" : [data[1]],
    "age" : [data[2]],
    "weight" : [data[3]],
    "hours" : [data[4]],
    "athmo_num" : [data[5]],
    "etat_surf_num" : [data[6]],
    "lum_num" : [data[7]],
    "agglo_num" : [data[8]]
  }
  test = pd.DataFrame(data=d)

  return_dict = {"gravity" : str(neigh.predict(test)[0])}

  json_object = json.dumps(return_dict, indent = 4) 

  return json_object


test = knn(sys.argv[2], data = sys.argv[1])

print(test)
