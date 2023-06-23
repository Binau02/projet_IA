import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



data = pd.read_csv("./data/stat_acc_V3.csv", sep=";")
data = pd.DataFrame(data)
data.drop(columns = ['Num_Acc', 'num_veh', 'id_usa', 'date', 'ville', 'id_code_insee', 'descr_cat_veh', 'descr_agglo', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'description_intersection', 'descr_dispo_secu', 'descr_grav', 'descr_motif_traj', 'descr_type_col', 'an_nais', 'place', 'dept', 'region', 'CODE_REG', 'weeks', 'month', 'days', 'intersection_num', 'motif_num', 'collision_num'], inplace = True)

x, y = data.drop(columns=["gravity"]), data.gravity

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

raf = RandomForestClassifier(max_depth=None, max_features='sqrt',n_estimators=1000)
raf.fit(x_train, y_train)
joblib.dump(raf, "./models/random_forest.joblib")