# Sample Usage:
# $./bin/spark-submit --driver-memory 4g ./samples/movie_recommender_system.py 

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import zipcode

PLOT_DATA = False

sc = SparkContext(appName="movie_recommender")
sc.setLogLevel("ERROR")

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"

DATAFILEPATH = PATH + "/data/ml-100k/u.data"
MOVIEFILEPATH = PATH + "/data/ml-100k/u.item"
USERFILEPATH = PATH + "/data/ml-100k/u.user"


if __name__ == '__main__':

	rawData =  sc.textFile(DATAFILEPATH)
	movieData = sc.textFile(MOVIEFILEPATH)
	userData = sc.textFile(USERFILEPATH)

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

	actual = []
	pred = []
	for index in range(len(results)):
		actual.append(results[index][1][0])
		pred.append(results[index][1][1])
		results[index]
		#print results[index], actual[index], pred[index]

	if PLOT_DATA:
		fig = plt.figure(figsize=(8,8))
		plt.plot(actual[:100], 'o', markersize=7, color='blue', alpha=0.5, label='actual')
		plt.plot(pred[:100], '^', markersize=7, color='green', alpha=0.5, label='predicted')
		plt.xlabel('Movie Index')
		plt.ylabel('Noramlized Ratings')
		plt.legend()
		plt.title('ALS Prediction')
		plt.show()

	# print "Model user features : ", model.userFeatures
	# for item in results[0:20]:
	# 	print item[0][0], "\t", item[0][1], "\t", item[1][0], "\t", item[1][1] 
	# print("Mean Squared Error Explicit: " + str(MSE))

	if len(sys.argv) < 3:
		userID = 771
		numProducts = 15
	else:
		userID = int(sys.argv[1])
		numProducts = int(sys.argv[2])

	#userInfo = userData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), int(fields[1]), fields[2], fields[3], int(fields[4]))).collect()
	userInfo = userData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), int(fields[1]), fields[2], fields[3], fields[4])).keyBy(lambda fields: fields[0])
	# collecting movie titles and associated movie itemIDs
	movieTitles = movieData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), fields[1])).collect()
	# creating RDD with movie titles and associated movie itemIDs keyed by itemIDs for looking up movies
	movieTitleList = movieData.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]), fields[1])).keyBy(lambda line:line[0])
	# capturing movie ratings sorted in descending order of rating value
	movieRatings = rawData.map(lambda line: line.split('\t')).map(lambda line: (int(line[0]), int(line[1]), float(line[2])/5.0)).sortBy(lambda line:-line[2])
	# keying ratings by users and looking up ratings from a specific user. This list is in descending order or rating scores
	moviesForUser = movieRatings.keyBy(lambda line:line[0]).lookup(userID)

	ratedTitles = []

	for item in moviesForUser:
		title = movieTitleList.lookup(item[1])
		ratedTitles.append(title)

	selectUserInfo = userInfo.lookup(userID)
	print 
	print selectUserInfo
	print "Age : ", selectUserInfo[0][1]
	print "Sex : ", selectUserInfo[0][2]
	print "Profession : ", selectUserInfo[0][3]
	try:
		ZipCode = int(selectUserInfo[0][4])
	except ValueError:
		#Handle the exception
		print 'Illegal ZipCode, setting to NYC **'
		ZipCode = 10001 #NYC

	print "ZipCode : ", ZipCode
	userZip = zipcode.isequal(str(ZipCode))
	print "Location : ", userZip.to_dict()		

	print
	print "Rated Products : "
	for index in range(numProducts):  
		if index < len(moviesForUser):
			print moviesForUser[index][1], ratedTitles[index][0][1]

	recommendedTitles = []
	topRecs = model.recommendProducts(userID, numProducts)

	for item in topRecs:
		title = movieTitleList.lookup(item[1])
		recommendedTitles.append(title)

	print
	print "Recommended Products : "
	for index in range(len(topRecs)):
		print topRecs[index][1], recommendedTitles[index][0][1]






