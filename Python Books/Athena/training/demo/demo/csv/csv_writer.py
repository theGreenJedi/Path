import csv

data = [('One', 1, 1.5), ('Two', 2, 8.0)]
f = open("out1.txt", "w")
writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_ALL)
writer.writerows(data)
f.close()