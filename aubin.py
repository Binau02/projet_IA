import pandas as pd
import math
import collections
import matplotlib.pyplot as plt
import time
import json





# find the best k -> k = 5

# arr_mean = []
# arr_best = []
# k = [*range(3, 16, 4)]
# for i in range(3, 16, 4):
#   print("k =", i)
#   sum_mean = 0
#   sum_best = 0
#   for j in range(30):
#     g = df.iloc[j]["gravity"]
#     test = knn('data/stat_acc_V3.csv', k=i, data = df.iloc[j].drop(labels = ["gravity"]).array)
#     sum_mean += abs(g-test[0])
#     if (g != test[1]):
#       sum_best += 1
#   arr_mean.append(sum_mean / 30)
#   arr_best.append(sum_best / 30)

# print(arr_mean)
# print(arr_best)

# plt.plot(k, arr_mean)
# plt.plot(k, arr_best)

# plt.show()
