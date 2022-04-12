# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql import types
from pyspark.sql.functions import *
from pyspark.sql.types import LongType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
from pyspark import StorageLevel

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer, MinMaxScaler, PCA
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.sklearn

import re
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import warnings

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# joined_full_null_imputed_pagerank_df.display(10)
model_transformed_train_2018_new_pagerank_noncontscale_df.display(10)

# COMMAND ----------

# joined_full_null_imputed_pagerank_df.printSchema()
model_transformed_train_2018_new_pagerank_noncontscale_df.printSchema()

# COMMAND ----------

model_transformed_train_2018_new_pagerank_noncontscale_df = model_transformed_train_2018_new_pagerank_noncontscale_df.na.drop()

# COMMAND ----------

# dataset_df = joined_full_null_imputed_pagerank_df.select(col("YEAR"), col("MONTH"), col("DEP_DEL15"), col("DEST_WEATHER_VIS_DISTANCE_DIMENSION"), col("pagerank"))
df = model_transformed_train_2018_new_pagerank_noncontscale_df.select(col("MONTH"),  col("DEP_DEL15"), col("pagerank"), col("PRIOR_DEP_DEL15"))

# COMMAND ----------

df.display(10)

# COMMAND ----------

list_80 = ['1', '2', '3', '4', '5', '6']
list_20 = ['7', '8']
train_df = df.where(col('MONTH').isin(list_80)).drop("MONTH") #1-6
test_df = df.where(col('MONTH') .isin(list_20)).drop("MONTH") #7-8
# train_df = df.where(col('MONTH') == '2').drop("MONTH") #1-6
# test_df = df.where(col('MONTH') == '3').drop("MONTH") #7-8

# COMMAND ----------

train_rdd = train_df.rdd
test_rdd = test_df.rdd

# COMMAND ----------

# this function converts each RDD rows into dataset used for model
def parse(row):
    """
    Map record_string --> (tuple,of,fields)
    """
    return((row[1], row[2]), row[0])

# COMMAND ----------

train_rdd = train_rdd.map(parse).cache()
test_rdd = test_rdd.map(parse).cache()

# COMMAND ----------

print(f"train count: {train_rdd.count()}")
print(f"test count: {test_rdd.count()}")

# COMMAND ----------

train_rdd.count()

# COMMAND ----------

test_rdd.collect()

# COMMAND ----------

# https://github.com/UCB-w261/main/blob/main/HelpfulResources/logistic_regression/Logistic%20regression.ipynb

# COMMAND ----------

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# COMMAND ----------

def LogLoss(dataRDD, W): 
    """
    Compute logistic loss error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    ################## YOUR CODE HERE ##################
    loss = augmentedData.map(lambda x: x[1]*np.log(sigmoid(W.dot(x[0]))) + (1-x[1])*np.log(1 - sigmoid(W.dot(x[0])))).mean()*-1
   
    ################## (END) YOUR CODE ##################
    return loss

# COMMAND ----------

def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one gradient descent step/update 
    Args:
        dataRDD - tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        learningRate - (float) defaults to 0.1
        regParam - (float) regularization term coefficient
    Returns:
        model   - (array) updated coefficients, bias still at index 0
    """
    # augmented data
    N=dataRDD.count()
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    
    new_model = None
    #################### YOUR CODE HERE ###################
    # Use the same way as before to find the first component of the gradient function
    # Note: the official equation should be a sum() which provides a lower log loss and a more accurate prediction.
    # However, Spark ML uses mean() for their calculation, possibly to avoid overflow issues.
        
    # Broadcast coeeficients
    sc.broadcast(W)  
    
    grad = augmentedData.map(lambda x: ((sigmoid(W.dot(x[0])) - x[1])*x[0])).sum()
#     if regType == 'ridge':
#         grad += regParam * np.append([0.0], W[1:])
#     elif regType == 'lasso':
#         grad += regParam * np.append([0.0], np.sign(W)[1:])
    new_model = W - learningRate * grad/N
    ################## (END) YOUR CODE ####################
    return new_model

# COMMAND ----------

# part d - ridge/lasso gradient descent function
def GradientDescent(trainRDD, testRDD, wInit, nSteps = 10, learningRate = 0.1, verbose = False):
    """
    Perform nSteps iterations of regularized gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps):  
        # update the model
        model = GDUpdate(trainRDD, model, learningRate)
        
        # keep track of test/train loss for plotting
        train_history.append(LogLoss(trainRDD, model))
        test_history.append(LogLoss(testRDD, model))
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history

# COMMAND ----------

wInit = np.random.uniform(0, 1, 3)
results = GradientDescent(train_rdd, test_rdd, wInit, nSteps = 100 )

# COMMAND ----------

# looking at loss of all iterations
results[0]

# Out[181]: [2.6189014073520847,
#  0.9484629531684111,
#  0.6911324047258114,
#  0.68392287776881,
#  0.67963398099526,
#  0.6762133436389024,
#  0.6730261062621423,
#  0.6699280011892952,
#  0.666891524242,
#  0.6639113940673458]

# COMMAND ----------

def getPrediction(testRDD, model): 
    augmentedData = testRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    
    predictions = augmentedData.map(lambda x: (float(sigmoid(model.dot(x[0]))), 1.0 if sigmoid(model.dot(x[0])) > .5 else 0.0, x[1]))
    
    return predictions

# COMMAND ----------

pred_result = getPrediction(test_rdd, results[2][9])

# COMMAND ----------

pred_result.take(10)

# 900 / 1000 = accuracy

# COMMAND ----------

pred_result.count()

# COMMAND ----------

# type(pred_result)


# COMMAND ----------

result = pred_result.map(lambda x: (x[1],x[2]))
metrics = MulticlassMetrics(result)
f05 = 1.25 * ((metrics.precision(1.0) * metrics.recall(1.0)) / ((0.25 * metrics.precision(1.0)) + metrics.recall(1.0)))
print(f"accuracy: {metrics.accuracy}")
print(f"Precision: {metrics.precision(1.0)}")
print(f"Recall: {metrics.recall(1.0)}")
print(f"FScore: {metrics.fMeasure(1.0, 0.5)}")
print(f"F0.5: {f05}")

conf_matrix = metrics.confusionMatrix().toArray()
print(f'TP = {conf_matrix[1][1]:.0f}')
print(f'FN = {conf_matrix[1][0]:.0f}')
print(f'TN = {conf_matrix[0][0]:.0f}')
print(f'FP = {conf_matrix[0][1]:.0f}')
print(f"Total Records: {conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0]:.0f}")
    
# accuracy = (result.sum()/result.count())*100
# result.collect()                

# print(result.take(3))

# month 2 and 3 and 0.5
# accuracy: 0.6411809569226398
# Precision: 0.1862466368201475
# Recall: 0.3272438443208896
# F0.5: 0.2038094170964847

# 80/20 
# accuracy: 0.7163956048089741
# Precision: 0.19283653551040453
# Recall: 0.11999251359775102
# F0.5: 0.1719583380772633

# COMMAND ----------

 print(metrics.confusionMatrix())

# COMMAND ----------

model_transformed_train_2015_18_new_pagerank_noncontscale_df.display(10)

# COMMAND ----------

metrics.confusion_matrix(actual, prediction)

# COMMAND ----------

cm = conf_matrix(actual, prediction, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=['business','health'])
cmd.plot()
cmd.ax_.set(xlabel='Predicted', ylabel='True')

# COMMAND ----------

prediction = result.map(lambda x: x[0])
actual = result.map(lambda x: x[1]) 
# prediction = [1, 0, 1, 0]
# actual = [1, 0, 0, 1]

# COMMAND ----------

prediction = prediction.collect()
actual = actual.collect()

# COMMAND ----------


from sklearn import metrics

# COMMAND ----------

