# Databricks notebook source
# load packages
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import datetime
import itertools

from pyspark.sql import Window, functions as f
from pyspark.sql.functions import col, to_timestamp, to_utc_timestamp, concat_ws, udf
from pyspark.sql import DataFrameNaFunctions as naf
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import RobustScaler, MinMaxScaler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

import random

from sklearn.metrics import confusion_matrix

from graphframes import *

plt.show(sns)

sc = spark.sparkContext

# set blob paths
blob_container = "w261-team28-container" 
storage_account = "team28" 
secret_scope = "w261-team28-scope"
secret_key = "w261-team28-key"
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# set data retrieval path
ROOT_PATH = 'dbfs:/mnt/mids-w261/datasets_final_project/'
ROOT_TEAM28_PATH = blob_url

# airline path
PARQ_AIRLINE_PATH = ROOT_PATH + 'parquet_airlines_data/'

# raw data path
AIRLINE_FULL_PATH = PARQ_AIRLINE_PATH + '201*.parquet'
AIRLINE_3M_PATH = ROOT_PATH + 'parquet_airlines_data_3m/*.parquet'
AIRLINE_6M_PATH = ROOT_PATH + 'parquet_airlines_data_6m/*.parquet'

# processed data path
AIRLINE_FULL_PROCESSED_PATH = blob_url + '/full_airline_data-processed'
AIRLINE_3M_PROCESSED_PATH = blob_url + '/processed/airline_3m_data.parquet'
AIRLINE_2015_PROCESSED_PATH = blob_url + '/processed/airline_2015_data.parquet'
AIRLINE_2019_PROCESSED_PATH = blob_url + '/airline_2019_data-processed'

WEATHER_2015_PROCESSED_PATH = blob_url + '/processed/weather_2015_data.parquet'
WEATHER_FULL_PROCESSED_PATH = blob_url + '/processed/weather_full.parquet'

JOINED_3M_PROCESSED_PATH = blob_url + '/processed/joined_3m_data.parquet'
JOINED_2015_PROCESSED_PATH = blob_url + '/processed/joined_2015_data.parquet'
JOINED_FULL_PROCESSED_PATH = blob_url + '/processed/joined_full_processed_df.parquet'
JOINED_FULL_NULL_IMPUTED_PROCESSED_PATH = blob_url + '/processed/joined_full_null_imputed_processed_df.parquet'
JOINED_FULL_NULL_IMPUTED_PAGERANK_PROCESSED_PATH = blob_url + '/processed/joined_full_null_imputed_processed_pagerank_df.parquet'

TRAIN_2015_PATH = blob_url + '/model/train_2015_data.parquet'
DEV_2015_PATH = blob_url + '/model/dev_2015_data.parquet'
TEST_2015_PATH = blob_url + '/model/test_2015_data.parquet'

TRAIN_2017_PATH = blob_url + '/model/train_2017_data.parquet'
VALID_2018_PATH = blob_url + '/model/valid_2018_data.parquet'
TEST_2019_PATH = blob_url + '/model/test_2019_data.parquet'

TRAIN_2017_NULL_IMPUTED_PATH = blob_url + '/model/train_2017_null_imputed_data.parquet'
VALID_2018_NULL_IMPUTED_PATH = blob_url + '/model/valid_2018_null_imputed_data.parquet'
TEST_2019_NULL_IMPUTED_PATH = blob_url + '/model/test_2019_null_imputed_data.parquet'

BASELINE_MODEL_2015_DF_PATH = blob_url + '/baseline_model_2015_df.parquet'

MODEL_TRANSFORMED_2017_18_CV_PATH_3 = blob_url + '/model_transformed_2017_18_CV_3.parquet'
MODEL_TRANSFORMED_FULL_CV_PATH = blob_url + '/model_transformed_full_CV.parquet'
MODEL_TRANSFORMED_TEST_CV_PATH = blob_url + '/model_transformed_test_CV.parquet'
MODEL_TRANSFORMED_FULL_CV_NEW_PATH = blob_url + '/model_transformed_full_CV_new.parquet'
MODEL_TRANSFORMED_TEST_CV_NEW_PATH = blob_url + '/model_transformed_test_CV_new.parquet'
MODEL_TRANSFORMED_FULL_CV_NEW_PAGERANK_PATH = blob_url + '/model_transformed_full_CV_new_pagerank.parquet'
MODEL_TRANSFORMED_TEST_CV_NEW_PAGERANK_PATH = blob_url + '/model_transformed_test_CV_new_pagerank.parquet'
MODEL_TRANSFORMED_FULL_CV_NEW_PAGERANK_NONCONTSCALE_PATH = blob_url + '/model_transformed_full_CV_new_pagerank_noncontscale.parquet'
MODEL_TRANSFORMED_TEST_CV_NEW_PAGERANK_NONCONTSCALE_PATH = blob_url + '/model_transformed_test_CV_new_pagerank_noncontscale.parquet'
MODEL_TRANSFORMED_2018_CV_NEW_PAGERANK_NONCONTSCALE_PATH = blob_url + '/model_transformed_2018_CV_new_pagerank_noncontscale.parquet'
MODEL_TRANSFORMED_2019_TEST_CV_NEW_PAGERANK_NONCONTSCALE_PATH = blob_url + '/model_transformed_2019_test_CV_new_pagerank_noncontscale.parquet'

MODEL_TRANSFORMED_TRAIN_2015_PATH = blob_url + '/model_transformed_train_2015.parquet'
MODEL_TRANSFORMED_TRAIN_2016_PATH = blob_url + '/model_transformed_train_2016.parquet'
MODEL_TRANSFORMED_TRAIN_2017_PATH = blob_url + '/model_transformed_train_2017.parquet'
MODEL_TRANSFORMED_TRAIN_2018_PATH = blob_url + '/model_transformed_train_2018.parquet'

MODEL_TRANSFORMED_VALID_2016_from_2015_PATH = blob_url + '/model_transformed_valid_2016_frm_2015.parquet'
MODEL_TRANSFORMED_VALID_2017_from_2015_PATH = blob_url + '/model_transformed_valid_2017_frm_2015.parquet'
MODEL_TRANSFORMED_VALID_2018_from_2015_PATH = blob_url + '/model_transformed_valid_2018_frm_2015.parquet'
MODEL_TRANSFORMED_VALID_2019_from_2015_PATH = blob_url + '/model_transformed_valid_2019_frm_2015.parquet'

MODEL_TRANSFORMED_VALID_2017_from_2016_PATH = blob_url + '/model_transformed_valid_2017_from_2016.parquet'
MODEL_TRANSFORMED_VALID_2018_from_2016_PATH = blob_url + '/model_transformed_valid_2018_from_2016.parquet'
MODEL_TRANSFORMED_VALID_2019_from_2016_PATH = blob_url + '/model_transformed_valid_2019_from_2016.parquet'

MODEL_TRANSFORMED_VALID_2018_from_2017_PATH = blob_url + '/model_transformed_valid_2018_from_2017.parquet'
MODEL_TRANSFORMED_VALID_2019_from_2017_PATH = blob_url + '/model_transformed_valid_2019_from_2017.parquet'

MODEL_TRANSFORMED_VALID_2019_from_2018_PATH = blob_url + '/model_transformed_valid_2019_from_2018.parquet'

ICAO_station_mapping_PATH = blob_url + '/ICAO_station_mapping' 

ICAO_station_mapping  = spark.read.parquet(ICAO_station_mapping_PATH).cache()

# read raw data
airline_full_raw_df = spark.read.parquet(AIRLINE_FULL_PATH).cache()
airline_3m_raw_df = spark.read.parquet(AIRLINE_3M_PATH).cache()
airline_6m_raw_df = spark.read.parquet(AIRLINE_6M_PATH).cache()

# read processed data
airline_full_processed_df = spark.read.parquet(AIRLINE_FULL_PROCESSED_PATH).cache()
airline_2015_processed_df = spark.read.parquet(AIRLINE_2015_PROCESSED_PATH).cache()
airline_2019_processed_df = spark.read.parquet(AIRLINE_2019_PROCESSED_PATH).cache()

weather_full_processed_df = spark.read.parquet(WEATHER_FULL_PROCESSED_PATH).cache()

joined_3m_processed_df = spark.read.parquet(JOINED_3M_PROCESSED_PATH).cache()
joined_2015_processed_df = spark.read.parquet(JOINED_2015_PROCESSED_PATH).cache()
joined_full_processed_df = spark.read.parquet(JOINED_FULL_PROCESSED_PATH).cache()
joined_full_null_imputed_df = spark.read.parquet(JOINED_FULL_NULL_IMPUTED_PROCESSED_PATH).cache()
joined_full_null_imputed_pagerank_df = spark.read.parquet(JOINED_FULL_NULL_IMPUTED_PAGERANK_PROCESSED_PATH).cache()

train_2015_df = spark.read.parquet(TRAIN_2015_PATH).cache()
dev_2015_df = spark.read.parquet(DEV_2015_PATH).cache()
test_2015_df = spark.read.parquet(TEST_2015_PATH).cache()

train_2017_df = spark.read.parquet(TRAIN_2017_PATH).cache()
valid_2018_df = spark.read.parquet(VALID_2018_PATH).cache()
test_2019_df = spark.read.parquet(TEST_2019_PATH).cache()

train_2017_null_imputed_df = spark.read.parquet(TRAIN_2017_NULL_IMPUTED_PATH).cache()
valid_2018_null_imputed_df = spark.read.parquet(VALID_2018_NULL_IMPUTED_PATH).cache()
test_2019_null_imputed_df = spark.read.parquet(TEST_2019_NULL_IMPUTED_PATH).cache()

# baseline dataset
baseline_model_2015_df = spark.read.parquet(BASELINE_MODEL_2015_DF_PATH).cache()

# read transformed data
model_transformed_2017_18_df = spark.read.parquet(MODEL_TRANSFORMED_2017_18_CV_PATH_3).cache()
model_transformed_train_2015_18_df = spark.read.parquet(MODEL_TRANSFORMED_FULL_CV_PATH).cache()
model_transformed_test_2019_df = spark.read.parquet(MODEL_TRANSFORMED_TEST_CV_PATH).cache()
model_transformed_train_2015_18_new_df = spark.read.parquet(MODEL_TRANSFORMED_FULL_CV_NEW_PATH).cache()
model_transformed_test_2019_new_df = spark.read.parquet(MODEL_TRANSFORMED_TEST_CV_NEW_PATH).cache()
model_transformed_train_2015_18_new_pagerank_df = spark.read.parquet(MODEL_TRANSFORMED_FULL_CV_NEW_PAGERANK_PATH).cache()
model_transformed_test_2019_new_pagerank_df = spark.read.parquet(MODEL_TRANSFORMED_TEST_CV_NEW_PAGERANK_PATH).cache()
model_transformed_train_2015_18_new_pagerank_noncontscale_df = spark.read.parquet(MODEL_TRANSFORMED_FULL_CV_NEW_PAGERANK_NONCONTSCALE_PATH).cache()
model_transformed_test_2019_new_pagerank_noncontscale_df = spark.read.parquet(MODEL_TRANSFORMED_TEST_CV_NEW_PAGERANK_NONCONTSCALE_PATH).cache()
model_transformed_train_2018_new_pagerank_noncontscale_df = spark.read.parquet(MODEL_TRANSFORMED_2018_CV_NEW_PAGERANK_NONCONTSCALE_PATH).cache()
model_transformed_test_2019frm2018_new_pagerank_noncontscale_df = spark.read.parquet(MODEL_TRANSFORMED_2019_TEST_CV_NEW_PAGERANK_NONCONTSCALE_PATH).cache()

model_transformed_train_2015_df = spark.read.parquet(MODEL_TRANSFORMED_TRAIN_2015_PATH).cache()
model_transformed_train_2016_df = spark.read.parquet(MODEL_TRANSFORMED_TRAIN_2016_PATH).cache()
model_transformed_train_2017_df = spark.read.parquet(MODEL_TRANSFORMED_TRAIN_2017_PATH).cache()
model_transformed_train_2018_df = spark.read.parquet(MODEL_TRANSFORMED_TRAIN_2018_PATH).cache()

model_transformed_valid_2016_from_2015_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2016_from_2015_PATH).cache()
model_transformed_valid_2017_from_2015_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2017_from_2015_PATH).cache()
model_transformed_valid_2018_from_2015_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2018_from_2015_PATH).cache()
model_transformed_test_2019_from_2015_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2019_from_2015_PATH).cache()

model_transformed_valid_2017_from_2016_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2017_from_2016_PATH).cache()
model_transformed_valid_2018_from_2016_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2018_from_2016_PATH).cache()
model_transformed_test_2019_from_2016_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2019_from_2016_PATH).cache()

model_transformed_valid_2018_from_2017_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2018_from_2017_PATH).cache()
model_transformed_test_2019_from_2017_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2019_from_2017_PATH).cache()

model_transformed_test_2019_from_2018_df = spark.read.parquet(MODEL_TRANSFORMED_VALID_2019_from_2018_PATH).cache()

# COMMAND ----------

