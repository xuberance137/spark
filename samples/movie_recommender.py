# Sample Usage:
# $./bin/spark-submit --driver-memory 4g ./samples/movie_recommender.py 

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sc = SparkContext(appName="data_visualization")
sc.setLogLevel("ERROR")

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"

DATAFILEPATH = PATH + "/data/ml-100k/u.data"
MOVIEFILEPATH = PATH + "/data/ml-100k/u.item"

rawData =  sc.textFile(DATAFILEPATH)
movieData = sc.textFile(MOVIEFILEPATH)

rawRatings = rawData.map(lambda line: line.split("\t")).take(3)

#user, movie, normalized rating -> Rating
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
# joining RDDs for predictions with original score resulting in {(user, item), (score, prediction)}
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# compute E((score-prediction)^2)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
results = ratesAndPreds.collect()

# # Build the recommendation model using Alternating Least Squares implicit (need to work on implicit factor generation)
# alpha = 0.01
# lambdax = 0.01
# model2 = ALS.trainImplicit(ratings, rank, numIterations, alpha)
# # predictions in form {(user, item), prediction}
# predictions = model2.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# # joining RDDs for predictions with original score resulting in {(user, item), (score, prediction)}
# ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# # compute E((score-prediction)^2)
# MSE2 = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# results2 = ratesAndPreds.collect()

print "Model user features : ", model.userFeatures
for item in results[0:20]:
	print item[0][0], "\t", item[0][1], "\t", item[1][0], "\t", item[1][1] 
print("Mean Squared Error Explicit: " + str(MSE))
# print("Mean Squared Error Implicit: " + str(MSE2))

userID = 771
numProducts = 15

# collecting movie titles and associated movie itemIDs
movieTitles = movieData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), fields[1])).collect()
# creating RDD with movie titles and associated movie itemIDs keyed by itemIDs for looking up movies
movieTitleList = movieData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), fields[1])).keyBy(lambda line:line[0])
# capturing movie ratings sorted in descending order
movieRatings = rawData.map(lambda line: line.split('\t')).map(lambda line: (int(line[0]), int(line[1]), float(line[2])/5.0)).sortBy(lambda line:-line[2])
# keying ratings by users and looking up ratings from a specific user. This list is in descending order or rating scores
moviesForUser = movieRatings.keyBy(lambda line:line[0]).lookup(userID)

ratedTitles = []

for item in moviesForUser:
	title = movieTitleList.lookup(item[1])
	ratedTitles.append(title)

print "Rated Products : "
for index in range(numProducts):  
	print moviesForUser[index][1], ratedTitles[index][0][1]

recommendedTitles = []
topRecs = model.recommendProducts(userID, numProducts)

for item in topRecs:
	title = movieTitleList.lookup(item[1])
	recommendedTitles.append(title)

print "Recommended Products : "
for index in range(len(topRecs)):
	print topRecs[index][1], recommendedTitles[index][0][1]






