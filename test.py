test = "[[1, 2], [3, 4], [5, 6]]"

test = test.replace(" ", '')
test = test.replace("[", '')
test = test[:-2]
test = test.split(sep = "],")
for i in range(len(test)):
  test[i] = test[i].split(sep = ",")
  test[i][0] = float(test[i][0])
  test[i][1] = float(test[i][1])

print(test)