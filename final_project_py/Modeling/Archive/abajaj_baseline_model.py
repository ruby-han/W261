# Databricks notebook source
# !pip install mlflow
# !pip install statsmodels


# COMMAND ----------



# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer, OneHotEncoder, VectorAssembler
import pyspark.sql.functions as f
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from statsmodels.tsa.seasonal import seasonal_decompose
# import mlflow.spark

# mlflow.pyspark.ml.autolog()
# # Prepare training documents from a list of (id, text, label) tuples.
# training = spark.createDataFrame([
#     (0, "a b c d e spark", 1.0),
#     (1, "b d", 0.0),
#     (2, "spark f g h", 1.0),
#     (3, "hadoop mapreduce", 0.0)
# ], ["id", "text", "label"])

# # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
# tokenizer = Tokenizer(inputCol="text", outputCol="words")
# hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
# lr = LogisticRegression(maxIter=10, regParam=0.001)
# pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# # Fit the pipeline to training documents.
# model = pipeline.fit(training)
# # Prepare test documents, which are unlabeled (id, text) tuples.
# test = spark.createDataFrame([
#     (4, "spark i j k"),
#     (5, "l m n"),
#     (6, "spark hadoop spark"),
#     (7, "apache hadoop")
# ], ["id", "text"])

# # Make predictions on test documents and print columns of interest.
# prediction = model.transform(test)
# selected = prediction.select("id", "text", "probability", "prediction")
# for row in selected.collect():
#     rid, text, prob, prediction = row  # type: ignore
#     print(
#         "(%d, %s) --> prob=%s, prediction=%f" % (
#             rid, text, str(prob), prediction   # type: ignore
#         )
#     )


# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import * # StringType
from pyspark.sql import Window
import pandas as pd
import datetime as dt
blob_container = "w261-team28-container" # The name of your container created in https://portal.azure.com
storage_account = "team28" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261-team28-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261-team28-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

display(dbutils.fs.ls(f"{mount_path}"))

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# joined_full_processed_df.display()
filtered_df_ts = joined_full_processed_df.withColumn('timestamp_unix',joined_full_processed_df.CRS_DEP_TIME_UTC.astype('Timestamp').cast("long")).cache()
w = Window.partitionBy('ORIGIN_AIRPORT_ID').orderBy('timestamp_unix').rangeBetween(-10800,-7200) # Between 2 & 3 hours

# COMMAND ----------

final_df = filtered_df_ts.withColumn('occurrences_in_60_min',f.avg('DEP_DEL15').over(w))
final_df.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Writing the feature to azure blob

# COMMAND ----------

final_df.write.mode('overwrite').parquet(f'{blob_url}/abajaj_feature/freq_delay_60_min.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ## DON'T GO ANY FURTHER `final_df` is the dataset with delay frequency per airport. The column with frequency is `occurrences_in_60_min`. There are some nulls.

# COMMAND ----------

df = spark.read.parquet(blob_url+"/processed/joined_2015_data.parquet").cache()
df.display()

# COMMAND ----------

# df.groupBy('ORIGIN_AIRPORT_ID').count().orderBy(col('count').desc()).display()
df.filter(df["ORIGIN_AIRPORT_ID"]=='13796').display()

# COMMAND ----------

# filtered_df = df.filter(df["TAIL_NUM"]=='N480HA').cache()

# COMMAND ----------

# df.select("CRS_DEP_TIME_UTC","ORIGIN_WEATHER_STATION_ID","DEP_DEL15").display()
filtered_df.groupby("ORIGIN_AIRPORT_ID").count().display()

# COMMAND ----------

# filtered_df = df.filter(df.ORIGIN_AIRPORT_ID == 72278023183).cache()

# COMMAND ----------

filtered_df_ts = filtered_df.withColumn('timestamp_unix',filtered_df.CRS_DEP_TIME_UTC.astype('Timestamp').cast("long")).cache()
w = Window.partitionBy('ORIGIN_AIRPORT_ID').orderBy('timestamp_unix').rangeBetween(-10800,-7200) # Between 2 & 3 hours
final_df = filtered_df_ts.withColumn('occurrences_in_60_min',f.avg('DEP_DEL15').over(w))


# COMMAND ----------

final_df.select("ORIGIN_AIRPORT_ID","CRS_DEP_TIME_UTC","occurrences_in_60_min","DEP_DEL15",'timestamp_unix').orderBy('timestamp_unix').display()

# COMMAND ----------

pd_df = filtered_df.select("CRS_DEP_TIME_UTC","ORIGIN_WEATHER_STATION_ID","DEP_DEL15").toPandas()#orderBy("CRS_DEP_TIME_UTC").display()

# COMMAND ----------

pd_df_copy = pd_df.copy(deep=True)

# COMMAND ----------

# pd_df['DEP_DEL15'].to_numpy()



# df = df.set_index(pd.DatetimeIndex(df['Datetime']))
# pd_df_copy = pd_df_copy.drop(['CRS_DEP_TIME_UTC','ORIGIN_WEATHER_STATION_ID'],axis=1)

analysis = pd_df_copy


decompose_result_mult = seasonal_decompose(analysis, model="additive",period=100)

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot();

# COMMAND ----------

df = spark.read.parquet(blob_url+"/processed/joined_2015_data.parquet").cache()
def fix_crs_dep_time(ts_str):
    try:
        ts_str = int(ts_str*0.01)
        return ts_str
    except Exception as e:
        raise(ts_str)
convertUDF = f.udf(lambda z: fix_crs_dep_time(z),IntegerType())
df = df.withColumn("CRS_DEP_TIME_HOUR", convertUDF(col("CRS_DEP_TIME")))
## BASELINE MODEL FEATURES
list_of_columns = set(df.columns)
TS_COLUMN = {'CRS_DEP_TIME_UTC_HOUR'}
PREDICTED = set(['DEP_DEL15'])
CONT_PREDICTORS =set(['AIRPORT_LAT_ORIGIN','AIRPORT_LONG_ORIGIN','ORIGIN_WEATHER_TMP_AIR_TEMP-AVG','ORIGIN_WEATHER_WND_SPEED_RATE-AVG'])
CAT_PREDICTORS = set(['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_DEP_TIME_HOUR'])
DROP_COLUMNS  = sorted(list(list_of_columns - PREDICTED.union(TS_COLUMN,CONT_PREDICTORS,CAT_PREDICTORS)))

# Transforming dataframe
transformer = Transformer(df)
transformer.drop_cols(DROP_COLUMNS)
transformer.drop_null()
transformer.one_hot_encode(list(CAT_PREDICTORS))
transformer.df = transformer.df.withColumn('SPLIT_ID', f.row_number().over(Window().orderBy(f.col('CRS_DEP_TIME_UTC_HOUR'))))

# train_df, test_df = DataSplit(df=transformer.df,train_percent=0.7,timestamp_col='CRS_DEP_TIME_UTC_HOUR').split_on_timestamp()
# train_df = train_df.drop('CRS_DEP_TIME_UTC_HOUR')
# test_df = test_df.drop('CRS_DEP_TIME_UTC_HOUR')

all_features_va = VectorAssembler(inputCols = ['AIRPORT_LAT_ORIGIN',
         'AIRPORT_LONG_ORIGIN',
         'CRS_DEP_TIME_HOUR_vec',
         'DAY_OF_MONTH_vec',
         'DAY_OF_WEEK_vec',
     'MONTH_vec','ORIGIN_WEATHER_TMP_AIR_TEMP-AVG', 'ORIGIN_WEATHER_WND_SPEED_RATE-AVG'], outputCol = 'ALL_FEATURES_VA')

transformer.df = all_features_va.transform(transformer.df)
# train_df = all_features_va.transform(train_df)
# test_df = all_features_va.transform(test_df)

# positive_df = train_df.filter(f.col('DEP_DEL15') == 1)
# positive_count = positive_df.count()
# negative_df = train_df.filter(f.col('DEP_DEL15') == 0)
# negative_count = negative_df.count()

# sample = 'undersample'
# fraction = positive_count/negative_count

# if sample == 'undersample':
#     train_df = negative_df.sample(withReplacement=False, 
#                                   fraction=fraction, 
#                                   seed = 1).unionAll(positive_df).cache()  
    
    
# Initializing model, runs & fits, and outputs a confusion matrix
# Model(train_df,test_df,label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'],
#                  **{
#                      'maxIter':30,
#                      'elasticNetParam':1,
#                      'regParam':0.01,
#                      'standardization':True
#                  }).get_logistic_regression()

# COMMAND ----------

transformer.df.printSchema()

# COMMAND ----------

BASELINE_MODEL_2015_DF_PATH = blob_url + '/baseline_model_2015_df.parquet'
transformer.df.write.mode('overwrite').parquet(BASELINE_MODEL_2015_DF_PATH)

# COMMAND ----------

Model(train_df,test_df,label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'],
                 **{
                     'maxIter':5,
                     'elasticNetParam':0.0,
                     'regParam':0.0,
                     'standardization':True
                 }).get_logistic_regression()

# COMMAND ----------

# transformer.df.printSchema()
transformer_df = transformer.df.withColumn('SPLIT_ID', f.row_number().over(Window().orderBy(f.col('CRS_DEP_TIME_UTC_HOUR'))))
# transformer.df.select(f.min('CRS_DEP_TIME_UTC_HOUR')).display() #.alias('min_ts')).take(1)[0]['min_ts']

# COMMAND ----------



# COMMAND ----------

transformer_df.printSchema()

# COMMAND ----------

FEATURE_SELECTED = set(['YEAR','QUARTER','MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','ORIGIN','DEST_AIRPORT_ID','DEST','DEP_DEL15','DEP_TIME_BLK',
'ARR_TIME_BLK','CRS_ELAPSED_TIME','ID','AIRPORT_LAT_ORIGIN','AIRPORT_LONG_ORIGIN','AIRPORT_LAT_DEST','AIRPORT_LONG_DEST','CRS_DEP_TIME_UTC_HOUR','CRS_DEP_TIME','ORIGIN_WEATHER_ELEVATION',
                        'ORIGIN_WEATHER_WND_DIRECTION_ANGLE-AVG','ORIGIN_WEATHER_WND_SPEED_RATE-AVG','ORIGIN_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG','ORIGIN_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
                      'ORIGIN_WEATHER_TMP_AIR_TEMP-AVG','ORIGIN_WEATHER_DEW_POINT_TEMP-AVG','ORIGIN_WEATHER_SLP_SEA_LEVEL_PRES-AVG','DEST_WEATHER_WND_DIRECTION_ANGLE-AVG','DEST_WEATHER_WND_SPEED_RATE-AVG','DEST_WEATHER_ELEVATION','DEST_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG','DEST_WEATHER_VIS_DISTANCE_DIMENSION-AVG','DEST_WEATHER_TMP_AIR_TEMP-AVG','DEST_WEATHER_DEW_POINT_TEMP-AVG','DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG'])

# COMMAND ----------

## BASELINE MODEL FEATURES
list_of_columns = set(df.columns)
TS_COLUMN = {'CRS_DEP_TIME_UTC_HOUR'}
PREDICTED = set(['DEP_DEL15'])
CONT_PREDICTORS =set(['AIRPORT_LAT_ORIGIN','AIRPORT_LONG_ORIGIN','ORIGIN_WEATHER_TMP_AIR_TEMP-AVG','ORIGIN_WEATHER_WND_SPEED_RATE-AVG'])
CAT_PREDICTORS = set(['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_DEP_TIME_HOUR'])
DROP_COLUMNS  = sorted(list(list_of_columns - PREDICTED.union(TS_COLUMN,CONT_PREDICTORS,CAT_PREDICTORS)))

# COMMAND ----------

class Transformer():
    
    def __init__(self,df):
        self.df = df
      
#     def one_hot(self,col_name):
#         pass
#         return df[categorical]
# cat = []

# for cat_feature in CAT_FEATURES:
#     cat_SI = StringIndexer(inputCol = cat_feature, outputCol = cat_feature + '_SI', handleInvalid = 'keep')
#     cat_OH = OneHotEncoder(inputCols = [cat_feature + '_SI'], outputCols = [cat_feature + '_OH'])
#     cat.extend([cat_SI, cat_OH])
# cat_VA = VectorAssembler(inputCols = [cat_feature + '_OH' for cat_feature in CAT_FEATURES], outputCol = 'CAT_FEATURES_VA')
# cat.extend([cat_VA])
    
    def time_based_transform(self,col_name='CRS_DEP_TIME_UTC',type_transform='hour',new_col_name = 'hour'):
        return self.df.withColumn(type_transform, eval("f."+type_transform)(col(col_name)))
    
    def filter_data(self,col_names_list):
        self.df =  self.df.select(*col_names_list) # GL added, feel free to change
    
    def drop_cols(self,col_names_list):
        self.df =  self.df.drop(*col_names_list)

    def drop_null(self,how='any',thresh=None,subset=None):
        self.df = self.df.dropna(how=how, thresh=None, subset=None)
#         self.df =  self.df.na.drop() # GL Edit based on testing 
    
    def one_hot_encode(self, col_names,col_type='categorical'):
        # TO DO: Fix this to take string type colums
#         if col_names.get('categorical',False):
#             StringIndexer(inputCol = cat_feature, outputCol = cat_feature + '_SI', handleInvalid = 'keep')
            
        outputcols = [col+'_vec' for col in col_names]
        one_hot_encode = OneHotEncoder(inputCols=col_names, outputCols=outputcols)
        self.df = one_hot_encode.fit(self.df).transform(self.df)
        self.drop_cols(col_names)
    
    def sampling(self):
        pass
    
class Model():
    
    def __init__(self,train_df, test_df, label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'], **kwargs):
        self.model_args = kwargs
        self.label_col = label_col
        self.feature_col = feature_col
        self.model = None
        self.train_df = train_df
        self.test_df = test_df
    
    def get_logistic_regression(self):
        lr = LogisticRegression(labelCol = self.label_col[0],
                                  featuresCol=self.feature_col[0],
                                  elasticNetParam = self.model_args.get('elasticNetParam',1),
                                  standardization = self.model_args.get('standardization',True),
                                  maxIter=self.model_args.get('maxIter',10), 
                                  regParam=self.model_args.get('regParam',0.001))
        
        self.model = lr.fit(self.train_df)
        self.get_predictions()
    
    def get_predictions(self):
#         tp = results[(results[self.feature_col[0]] == 1) & (results[self.feature_col[0]] == 1)].count()
#         tn = results[(results[self.feature_col[0]] == 0) & (results[self.feature_col[0]] == 0)].count() 
#         fp = results[(results[self.feature_col[0]] == 0) & (results[self.feature_col[0]] == 1)].count()
#         fn = results[(results[self.feature_col[0]] == 1) & (results[self.feature_col[0]] == 0)].count()
#         print("True Positives:", tp)
#         print("True Negatives:", tn)
#         print("False Positives:", fp)
#         print("False Negatives:", fn)
#         print("Total", results.count())
        results = self.model.transform(self.test_df)
        preds_train = self.model.transform(self.train_df)
        pandas_df = results.select([self.label_col[0],'prediction']).toPandas()
        cf_matrix = confusion_matrix(y_true=pandas_df[self.label_col], y_pred=pandas_df['prediction'],normalize='true')
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['True','False'])
        self.get_metrics(preds_train,results)
    
    def get_metrics(self,preds_train, preds_test):
        
        train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
        train_metrics = MulticlassMetrics(train_rdd)
        valid_rdd = preds_test.select(['prediction', 'DEP_DEL15']).rdd
        valid_metrics = MulticlassMetrics(valid_rdd)

        print('BASELINE LOGISTIC REGRESSION')
        print(' \t\tTrain Metrics \t Validation Metrics\n')
        print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
        print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
        print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
        print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
        print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
        print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')
        

class DataSplit():
    
    def __init__(self,df,train_percent=0.8,timestamp_col='CRS_DEP_TIME_UTC'):
        self.df = df
        self.train_percent = train_percent
        self.test_precent = 1 - train_percent
        self.col_name = timestamp_col
    
    def split_on_timestamp(self):
        
        min_ts = self.df.select(f.min(self.col_name).alias('min_ts')).take(1)[0]['min_ts']
        max_ts = self.df.select(f.max(self.col_name).alias('max_ts')).take(1)[0]['max_ts']
        time_range = (max_ts - min_ts).days
        train_threshold_days = min_ts + dt.timedelta(days=int(self.train_percent*time_range))
        df_train = self.df.filter(col(self.col_name) < train_threshold_days)
        df_test = self.df.filter(col(self.col_name) >= train_threshold_days)
        
        return [df_train, df_test]

    def cv_folds(self, data = None, folds = 1, train_pct = .8, simple_splits = False):
        # Empty list to store dataframes for each fold
        train_dfs = []
        val_dfs = []

        # Create simple folds by full train year, val year
        if simple_splits:
            for year in [2015, 2016, 2017]:

                train_df = data.where(f.col('CRS_DEP_TIME_UTC_HOUR') == year)
                val_df = data.where(f.col('CRS_DEP_TIME_UTC_HOUR') == (year + 1))

                train_dfs.append(train_df)
                val_dfs.append(val_df)

        # Create folds by number of folds, train/test split %
        else:
            data_size = data.count()

            for k in range(folds): 

                # Calculate total size of each fold
                fold_size = data_size/folds

                # Find range of `SPLIT_ID` for train and val
                train_ids = ((fold_size * k) + 0, 
                             int(fold_size * k + fold_size*.8))

                val_ids = (train_ids[1] + 1, 
                           fold_size * k + fold_size)            

                # Split the data
                train_df = data.where(f.col('SPLIT_ID').between(train_ids[0], train_ids[1]))
                val_df = data.where(f.col('SPLIT_ID').between(val_ids[0], val_ids[1]))

                # store data
                train_dfs.append(train_df)
                val_dfs.append(val_df)

        return train_dfs, val_dfs
        
        


# COMMAND ----------

# Transforming dataframe
transformer = Transformer(df)
transformer.drop_cols(DROP_COLUMNS)
transformer.drop_null()
transformer.one_hot_encode(list(CAT_PREDICTORS))
transformer.df.display()

# COMMAND ----------

# Getting Train & Test
train_df, test_df = DataSplit(df=transformer.df,train_percent=0.7,timestamp_col='CRS_DEP_TIME_UTC_HOUR').split_on_timestamp()
train_df = train_df.drop('CRS_DEP_TIME_UTC_HOUR')
test_df = test_df.drop('CRS_DEP_TIME_UTC_HOUR')

# COMMAND ----------

all_features_va = VectorAssembler(inputCols = ['AIRPORT_LAT_ORIGIN',
         'AIRPORT_LONG_ORIGIN',
         'CRS_DEP_TIME_HOUR_vec',
         'DAY_OF_MONTH_vec',
         'DAY_OF_WEEK_vec',
     'MONTH_vec','ORIGIN_WEATHER_TMP_AIR_TEMP-AVG', 'ORIGIN_WEATHER_WND_SPEED_RATE-AVG'], outputCol = 'ALL_FEATURES_VA')

train_df = all_features_va.transform(train_df)
test_df = all_features_va.transform(test_df)

# COMMAND ----------

positive_df = train_df.filter(f.col('DEP_DEL15') == 1)
positive_count = positive_df.count()
negative_df = train_df.filter(f.col('DEP_DEL15') == 0)
negative_count = negative_df.count()

sample = 'undersample'
fraction = positive_count/negative_count

if sample == 'undersample':
    train_df = negative_df.sample(withReplacement=False, 
                                  fraction=fraction, 
                                  seed = 1).unionAll(positive_df).cache()  

# COMMAND ----------

# Initializing model, runs & fits, and outputs a confusion matrix

Model(train_df,test_df,label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'],
                 **{
                     'maxIter':30,
                     'elasticNetParam':1,
                     'regParam':0.01,
                     'standardization':True
                 }).get_logistic_regression()

# COMMAND ----------

# MAX_ITER = 30
# ALPHA = 1 # 0.1
# LAMBDA = 0.01

# log_reg = LogisticRegression(labelCol = 'DEP_DEL15', 
#                             featuresCol = 'ALL_FEATURES_VA', 
#                             maxIter = MAX_ITER, 
#                             elasticNetParam = ALPHA,
#                             regParam = LAMBDA, standardization=True)

# log_reg_model = log_reg.fit(train_df)
# results = log_reg_model.transform(test_df)
# metricsp = MulticlassMetrics(results.select(['DEP_DEL15','prediction']).rdd)
# # metricsp.recall(1)
# tp = results[(results.DEP_DEL15 == 1) & (results.prediction == 1)].count()
# tn = results[(results.DEP_DEL15 == 0) & (results.prediction == 0)].count()
# fp = results[(results.DEP_DEL15 == 0) & (results.prediction == 1)].count()
# fn = results[(results.DEP_DEL15 == 1) & (results.prediction == 0)].count()
# print("True Positives:", tp)
# print("True Negatives:", tn)
# print("False Positives:", fp)
# print("False Negatives:", fn)
# print("Total", results.count())
# my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='DEP_DEL15')
# results.select('DEP_DEL15','prediction')
# AUC = my_eval.evaluate(results)
# print("AUC score is : ",AUC)


# COMMAND ----------

# # results.select(['DEP_DEL15','prediction']).rdd.take(5)#()#display()
# # metricsp = MulticlassMetrics(results.select(['DEP_DEL15','prediction']).rdd)
# # metricsp.confusionMatrix()#.collect()#toArray()
# # cf_matrix = confusion_matrix(y_test, y_pred)

# # pandas_df = results.select(['DEP_DEL15','prediction']).toPandas()
# cf_matrix = confusion_matrix(y_true=pandas_df['DEP_DEL15'], y_pred=pandas_df['prediction'])

# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# ax.set_title('Confusion Matrix with labels\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['True','False'])

# ## Display the visualization of the Confusion Matrix.
# # plt.show()

# COMMAND ----------

# preds_train.select(["DEP_DEL15","prediction"]).distinct().display()
# preds_train.select(["prediction"],f.countDistinct(col("prediction"))).display()
# preds_train.select(countDistinct("DEP_DEL15","prediction")).display()
# preds_train.groupBy('prediction').count().display()

# COMMAND ----------

# train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
# train_metrics = MulticlassMetrics(train_rdd)

# COMMAND ----------

# print(train_metrics.recall(label=0))
# print(train_metrics.precision(0))
# print(train_metrics.accuracy)
# print(train_metrics.fMeasure(1.0))

# COMMAND ----------

# from pyspark.ml.feature import OneHotEncoder, VectorAssembler

# df = spark.createDataFrame([
#     (0.0, 1.0),
#     (1.0, 0.0),
#     (2.0, 1.0),
#     (0.0, 2.0),
#     (0.0, 1.0),
#     (2.0, 0.0)
# ], ["categoryIndex1", "categoryIndex2"])
# df.display()
# encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
#                         outputCols=["categoryVec1", "categoryVec2"])
# model = encoder.fit(df)
# encoded = model.transform(df)
# encoded.show()#.toPandas()
# VectorAssembler(inputCols =['categoryVec1','categoryVec2'], outputCol = 'categoryVec_VA').transform(encoded).show()
# # encoded.show()

# COMMAND ----------

