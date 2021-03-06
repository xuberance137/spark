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

def squared_error(actual, pred):
	return (pred - actual)**2

def abs_error(actual, pred):
	return np.abs(pred - actual)

def squared_log_error(actual, pred):
	return (np.log(1+pred) - np.log(1+actual))**2

def evaulate(train, test, iterations, step, regParam, regType, intercept):
	model = LinearRegressionWithSGD.train(train, iterations=iterations, step=float(step), intercept=intercept)
	tp = test.map(lambda p: (p.label, model.predict(p.features)))
	rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
	return rmsle


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

# formating RDDs from processed data to LabelPoints
data = records.map(lambda datapoint: LabeledPoint(extract_label(datapoint), extract_features(datapoint)))
data_dt = records.map(lambda datapoint: LabeledPoint(extract_label(datapoint), extract_features_dt(datapoint)))
# training model and predicting for in sample LabelPoints
linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))

dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data_dt.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)
# variable transformation. Using log domain representation of target variable
data_log = data.map(lambda p: LabeledPoint(np.log(p.label), p.features))
model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)
true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label), model_log.predict(p.features)))

# Performanace Metrics
mae = true_vs_predicted.map(lambda (actual, pred): abs_error(actual, pred)).mean()
rmse = np.sqrt(true_vs_predicted.map(lambda (actual, pred): squared_error(actual, pred)).mean())
rmsle = np.sqrt(true_vs_predicted.map(lambda (actual, pred): squared_log_error(actual, pred)).mean())

mae_dt = true_vs_predicted_dt.map(lambda (actual, pred): abs_error(actual, pred)).mean()
rmse_dt = np.sqrt(true_vs_predicted_dt.map(lambda (actual, pred): squared_error(actual, pred)).mean())
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (actual, pred): squared_log_error(actual, pred)).mean())

mae_log = true_vs_predicted_log.map(lambda (actual, pred): abs_error(actual, pred)).mean()
rmse_log = np.sqrt(true_vs_predicted_log.map(lambda (actual, pred): squared_error(actual, pred)).mean())
rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda (actual, pred): squared_log_error(actual, pred)).mean())

# print "Linear Model predictions : "
# for item in true_vs_predicted.take(10):
# 	print item

# print "Decision Tree predictions : "
# for item in true_vs_predicted_dt.take(10):
# 	print item

print "Linear Model MAE : ", mae, " RMSE : ", rmse, " RMSLE : ", rmsle
print "Decision Tree MAE : ", mae_dt, " RMSE : ", rmse_dt, " RMSLE : ", rmsle_dt
print "Log Linear Model MAE : ", mae_log, " RMSE : ", rmse_log, " RMSLE : ", rmsle_log

# Creating training and testing data sets
data_with_index = data.zipWithIndex().map(lambda (k,v): (v,k))
data_test = data_with_index.sample(False, 0.2, 42).map(lambda (index, p): p)
data_train = data_with_index.subtractByKey(data_test).map(lambda (index, p): p)

data_with_index_dt = data_dt.zipWithIndex().map(lambda (k,v): (v,k))
data_test_dt = data_with_index.sample(False, 0.2, 42).map(lambda (index, p): p)
data_train_dt = data_with_index.subtractByKey(data_test).map(lambda (index, p): p)

test_val = evaulate(data_train, data_test, 5, 0.01, 0.0, '12', False)
print test_val

# params = [1,5,10,20, 50, 100]
# metrics = [evaulate(data_train, data_test, param, 0.01, 0.0, '12', False) for param in params]
# print params
# print metrics





















