# Sample Usage:
# $./bin/spark-submit --driver-memory 4g ./samples/movie_recommender.py 

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sc = SparkContext(appName="data_visualization")

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"

DATAFILEPATH = PATH + "/data/ml-100k/u.data"

rawData =  sc.textFile(DATAFILEPATH)

rawRatings = rawData.map(lambda line: line.split("\t")).take(3)

#user, movie, rating -> Rating
ratings = rawData.map(lambda line: line.split('\t')).map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])/5.0))

# Build the recommendation model using Alternating Least Squares
rank = 50
numIterations = 10 #for some reason crashes when this number is increased beyond 10
reg_param = 0.01
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
# predictions in form {(user, item), prediction}
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# joining RDDDs for predictions with original score resulting in {(user, item), (score, prediction)}
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# compute E((score-prediction)^2)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

results = ratesAndPreds.collect()

print "Model user features : ", model.userFeatures
for item in results:
	print item[0][0], "\t", item[0][1], "\t", item[1][0], "\t", item[1][1] 
print("Mean Squared Error : " + str(MSE))

userID = 771
numProducts = 15
topRecs = model.recommendProducts(userID, numProducts)
print "Recommended Products : ", topRecs
