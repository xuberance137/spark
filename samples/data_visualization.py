# Sample Usage:
# $bin/spark-submit ./test/data_visualization.py

from pyspark import SparkConf
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sc = SparkContext(appName="data_visualization")

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"

DATAFILEPATH = PATH + "/data/ml-100k/u.user"
userdata = sc.textFile(DATAFILEPATH)

userfields = userdata.map(lambda line: line.split("|"))
num_users = userfields.map(lambda fields: fields[0]).count()
num_occupations = userfields.map(lambda fields: fields[3]).distinct().count()

ages = userfields.map(lambda fields: fields[1]).collect()
num_bins = 20

print ages

# f = open('./examples/src/main/python/data/ages.txt', 'w')
# f.writelines(["%s\n" % item  for item in ages])
# f.close()

data_numeric = []
for item in ages:
	data_numeric.append(int(item))

# sns.set(color_codes=True)
# sns.distplot(data_numeric, bins=num_bins, rug=True, label="Ages", rug_kws={"color": "r"})
# sns.plt.show()

count_by_occupation = userfields.map(lambda fields: (fields[3], 1)).reduceByKey(lambda x,y : x+y).collect()

xaxis1 = np.array([c[0] for c in count_by_occupation])
yaxis1 = np.array([c[1] for c in count_by_occupation])

xaxis = xaxis1[np.argsort(yaxis1)]
yaxis = yaxis1[np.argsort(yaxis1)]

pos = np.arange(len(xaxis))
width = 1.0

DATAFILEPATH = PATH + "/data/ml-100k/u.data"
ratingdataraw = sc.textFile(DATAFILEPATH)
ratingdata = ratingdataraw.map(lambda line: line.split("\t"))
ratings = ratingdata.map(lambda line: int(line[2]))
count = ratings.count()
min_rating = ratings.reduce(lambda x, y : min(x, y)) #combining elements in parallel
max_rating = ratings.reduce(lambda x, y : max(x, y))
mean_rating = ratings.reduce(lambda x, y : x+y ) / count
median_rating = np.median(ratings.collect()) 
rated_users = [x[0] for x in ratingdata.map(lambda line: line[0]).countByValue()]
num_rated_users = len(rated_users)



print "Count : %d" %count
print "Min rating : %d" % min_rating
print "Max rating : %d" % max_rating
print "Mean rating : %2.5f" % mean_rating
print "median rating : %d" % median_rating
print "Num rated users : %d" %num_rated_users

user_ratings_user = ratingdata.map(lambda fields: (int(fields[0]), int(fields[2]))).groupByKey().map(lambda (x, y): (x, len(y)))

user_list = user_ratings_user.take(20)

print user_list


sns.set_style("darkgrid")
ax = plt.axes()
ax.set_xticks(pos + (width/2))
ax.set_xticklabels(xaxis)
plt.bar(pos, yaxis, width, color='red')
plt.xticks(rotation = 90)
plt.show()





