# Sample Usage:
# $./bin/spark-submit ./samples/sharing_predictor.py 

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sc = SparkContext(appName="sharing_predictor")
sc.setLogLevel("ERROR")

def get_mapping(rdd, idx):
	return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

#mapping categorialfeatures to binary vectors
def extract_features(record):
	cat_vec = np.zeros(cat_len)
	i = 0
	step = 0
	for field in record[2:9]:
		m = mappings[i]
		index = m[field]
		cat_vec[index + step] = 1
		i = i + 1
		step = step + len(m)
	num_vec = np.array([float(field) for field in record[10:14]])

	return np.concatenate((cat_vec, num_vec))

def extract_label(record):
	return float(record[-1])

def extract_features_dt(record):
	return np.array(map(float, record[2:14]))


PATH = "/Users/gopal/projects/learning/spark/spark-1.6.1-bin-hadoop2.6"
DATAFILEPATH = PATH + "/data/Bike-Sharing-Dataset/hour_noheader.csv"

rawData = sc.textFile(DATAFILEPATH)
numData = rawData.count()
records = rawData.map(lambda x: x.split(","))
records.cache()
first = records.first()

#run get_mapping function over categorical features
mappings = [get_mapping(records, index) for index in range(2, 10)]
#length of categorial features mapped to binary vectors
cat_len = sum(map(len, mappings))
# length of numeric features
# print records.first()[10:14]
num_len = len(records.first()[10:14])
# length of linear regressor feature vector
total_len =  cat_len + num_len

data = records.map(lambda datapoint: LabeledPoint(extract_label(datapoint), extract_features(datapoint)))
data_dt = records.map(lambda datapoint: LabeledPoint(extract_label(datapoint), extract_features_dt(datapoint)))

# sample = data.first()
# print "Label : ", str(sample.label)
# print "Features : ", str(sample.features)

linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))

dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data_dt.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)

print "Linear Model predictions : "
for item in true_vs_predicted.take(10):
	print item

print "Decision Tree predictions : "
for item in true_vs_predicted_dt.take(10):
	print item







