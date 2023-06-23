import sys
import joblib
import pandas as pd
import json



def highLevel(data, model):
  if type(data) == type(""):
    data = data.replace(' ', '')
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split(',')

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

  load_model = joblib.load("./models/" + model + ".joblib")

  return_dict = {"gravity" : str(load_model.predict(test)[0])}
  json_object = json.dumps(return_dict, indent = 4)
  return json_object

json = highLevel(sys.argv[1], sys.argv[2])

print(json)
