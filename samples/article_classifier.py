# Sample Usage:
# $./bin/spark-submit --driver-memory 4g ./samples/movie_recommender.py 

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import DecisionTree
#from pyspark.mllib.tree.configuration import Algo
from pyspark.mllib.tree.impurity import Entropy

numIterations = 10
maxTreeDepth = 5

sc = SparkContext(appName="article_classifier")
sc.setLogLevel("ERROR")

# feature values come in as a string. Replace ? and blanks with 0.0f, negative numbers with 0.0f 
def cleanup(x):
	a = []
	for item in x:
		if item == "?":
			item = 0.0
		if item is None:
			item = 0.0
		item = float(item)
		if item < 0.0:
			item = 0.0
		a.append(item)

	return a

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"

DATAFILEPATH = PATH + "/data/stumbleupon/train_noheader.tsv"

rawData =  sc.textFile(DATAFILEPATH)

rawColumns = rawData.map(lambda line: line.replace('\"', '')).map(lambda line: line.split("\t"))
alchemyCategories = rawColumns.map(lambda line: (line[3], line[4])).sortBy(lambda line: line[0]).collect()
label = rawColumns.map(lambda line: int(line[len(line)-1]))
features = rawColumns.map(lambda line: line[4:len(line)-1]).map(cleanup) #.take(500)
cleanData = rawColumns.map(lambda line: line[4:len(line)]).map(cleanup) #.take(500)
parsedData = cleanData.map(lambda line: LabeledPoint(int(line[len(line)-1]), line[0:len(line)-1]))

#print parsedData

lrModel = LogisticRegressionWithSGD.train(parsedData, numIterations)
svmModel = SVMWithSGD.train(parsedData, numIterations)
nbModel = NaiveBayes.train(parsedData)
dtModel = DecisionTree.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)



