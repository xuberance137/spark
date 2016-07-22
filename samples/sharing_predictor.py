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

sc = SparkContext(appName="sharing_predictor")
sc.setLogLevel("ERROR")

def get_mapping(rdd, idx):
	return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"
DATAFILEPATH = PATH + "/data/Bike-Sharing-Dataset/hour_noheader.csv"

rawData = sc.textFile(DATAFILEPATH)
numData = rawData.count()
records = rawData.map(lambda x: x.split(","))
records.cache()
first = records.first()

#new_rdd = get_mapping(records, 2)

#run get_mapping function over categorical features
mappings = [get_mapping(records, index) for index in range(2, 10)]
print mappings

#length of categorial features mapped to binary vectors
cat_len = sum(map(len, mappings))
#length of numeric features
print records.first()[10:14]
num_len = len(records.first()[10:14])
total_len =  cat_len + num_len


