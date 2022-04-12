# Databricks notebook source
# MAGIC %md
# MAGIC # Weather Data Processing
# MAGIC 
# MAGIC The purpose of this notebook is to organize all of the data processing actions that were taken throughout the EDA flow. The final output is a parquet file that will be uploaded to the blob for usage in transformations

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Data
# MAGIC Team note - no real changes here yet from starter_nb - minimizing

# COMMAND ----------

# Load Packages |

from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql.functions import col, to_timestamp, to_utc_timestamp, concat_ws, udf
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import airporttime

# COMMAND ----------

# Set Blob paths, create links | 

blob_container = "w261-team28-container" # The name of your container created in https://portal.azure.com
storage_account = "team28" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261-team28-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261-team28-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Data Size function | 

DATA_PATH = 'dbfs:/mnt/mids-w261/datasets_final_project/'

def deep_ls(path: str):
    """List all files in base path recursively."""
    tot = 0
    for x in dbutils.fs.ls(path):
        if x.path[-1] != '/':
            tot += x.size
            yield x
        else:
            for y in deep_ls(x.path):
                yield y
    yield f'DATASIZE: {tot}'

# print(*deep_ls(DATA_PATH), sep='\n')

total_size = []
for i in deep_ls(DATA_PATH):
  if 'DATASIZE:' in i:
    total_size.append(int(i.split(' ')[1]))

# COMMAND ----------

# Inspect the Mount's Final Project folder | 

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/"))
# display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/airlines_data/2015/"))
# display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/"))

# COMMAND ----------

# Load datasets into spark | 

datasets_final_project_path = 'dbfs:/mnt/mids-w261/datasets_final_project'
parquet_airlines_data_path = datasets_final_project_path + '/parquet_airlines_data/'
parquet_airlines_data_3m_path = datasets_final_project_path + '/parquet_airlines_data_3m/'
parquet_airlines_data_6m_path = datasets_final_project_path + '/parquet_airlines_data_6m/'
stations_data_path = datasets_final_project_path + '/stations_data/'
weather_data_path = datasets_final_project_path + '/weather_data/'

df_airlines_data_3m = spark.read.parquet(parquet_airlines_data_3m_path + '*.parquet').cache() # 2015 Q1 flights
df_airlines_data_6m = spark.read.parquet(parquet_airlines_data_6m_path + '*.parquet').cache() # 2015 Q1+Q2 flights

# main_data = df_airlines_data_3m

# COMMAND ----------

# df_weather = spark.read.parquet(f'/mnt/mids-w261/datasets_final_project/weather_data/*').filter(col('DATE') < "2015-04-01T00:00:00.000")
# df_weather = spark.read.parquet(f'/mnt/mids-w261/datasets_final_project/weather_data/*').filter(f.col('DATE') < "2016-01-01T00:00:00.000")
df_weather = spark.read.parquet(f'/mnt/mids-w261/datasets_final_project/weather_data/*')
# df_weather_new = spark.read.parquet(f'/mnt/mids-w261/datasets_final_project/weather_data/*').filter((f.col('DATE') > "2015-02-20T23:59:59.000") & (f.col('DATE') < "2015-02-22T00:00:00.000"))

# df_weather_subset = df_weather.sample(False, 0.01, seed = 1)
# df_weather_subset.write.parquet(f'{blob_url}/weather_sample_300k')
# df_weather_subset = spark.read.parquet(f'{blob_url}/weather_sample_300k').cache()

# df_weather_subset = spark.read.parquet(f'{blob_url}/weather_sample_30k').cache() # uncomment this

# COMMAND ----------

# df_weather_new.count()
# df_weather_new.cache()
df_weather_subset = df_weather # Assigning to name to run below code

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Reduce to relevant features, reduce to US weather-stations

# COMMAND ----------

FEATURES = eval(dbutils.notebook.run('/Users/rubyhan@berkeley.edu/team28/Final_Project/featureSelection', 60))

# Keep selected features
weather_full = df_weather_subset.select(FEATURES['WEATHER'])

# Keep US weather stations
df_weather_US = weather_full.filter(f.col('NAME').endswith('US'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dealing with Null values in `STATION` and `NAME`
# MAGIC Before we can move on to EDA or aggregations, we have to properly treat `STATION` and `NAME`. If we have null values in these columns, it will disallow proper aggregation or joins

# COMMAND ----------

# First, create a mapping table that take all non-null station ids and maps them to all non-null station names:
name_to_station = df_weather_US.where((col('NAME').isNull() == False) & (col('STATION').isNull() == False))\
                               .select('NAME', 'STATION')\
                               .withColumnRenamed('STATION', 'REP_STATION')\
                               .withColumnRenamed('NAME', 'REP_NAME')\
                               .distinct()

# First, need to focus on replacing null weather stations with known station id. we do this by creating a new column of non-null stations via join, then replacing station with rep_station (which is non-null)
df_weather_US_temp = df_weather_US.join(name_to_station, df_weather_US.NAME == name_to_station.REP_NAME, "left")\
                                  .select(df_weather_US['*'], name_to_station['REP_STATION'])\
                                  .withColumn('STATION', f.when(f.col('STATION').isNull(), col('REP_STATION'))\
                                                              .otherwise(f.col('STATION')))\
                                                              .drop('REP_STATION')

# Second, need to focus on replacing null weather stations with known name. we do this by creating a new column of non-null names via join, then replacing station with rep_name (which is non-null)
df_weather_US_temp = df_weather_US_temp.join(name_to_station, df_weather_US_temp.STATION == name_to_station.REP_STATION, "left")\
                                  .select(df_weather_US_temp['*'], name_to_station['REP_NAME'])\
                                  .withColumn('NAME', f.when(f.col('NAME').isNull(), col('REP_NAME'))\
                                                              .otherwise(f.col('NAME')))\
                                                              .drop('REP_NAME')


# display(df_weather_US_temp)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aggregating weather data by hour

# COMMAND ----------

df_weather_h = df_weather_US_temp.withColumn('HOUR', f.date_trunc('hour', f.col('DATE')))\
                            .withColumn('ROW_NUM', f.row_number().over(Window.partitionBy('STATION', 'HOUR')\
                                                                 .orderBy(col('DATE').desc())))\
                            .filter(f.col('ROW_NUM') == 1)\
                            .drop('ROW_NUM')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Break down concattenated features into distinct columns

# COMMAND ----------

# Decoding weather condition from NOAA
# https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# Wind params page 8-9
# Cloud params page 9
# Visibility params page 10
# Temperature page 10-11
# Dew point temperature page 11
# Pressure page 12

def weather_decode(df):
  WND = f.split(df['WND'], ',')
  CIG = f.split(df['CIG'], ',')
  VIS = f.split(df['VIS'], ',')
  TMP = f.split(df['TMP'], ',')
  DEW = f.split(df['DEW'], ',')
  SLP = f.split(df['SLP'], ',') 
  df = (df
        # WND
        .withColumn('WND_DIRECTION_ANGLE', WND.getItem(0).cast('int')) # numeric
        .withColumn('WND_QUALITY_CODE', WND.getItem(1).cast('int')) # categorical
        .withColumn('WND_TYPE_CODE', WND.getItem(2).cast('string')) # categorical
        .withColumn('WND_SPEED_RATE', WND.getItem(3).cast('int')) # categorical
        .withColumn('WND_SPEED_QUALITY_CODE', WND.getItem(4).cast('int')) # categorical
        
        # CIG
        .withColumn('CIG_CEILING_HEIGHT_DIMENSION', CIG.getItem(0).cast('int')) # numeric 
        .withColumn('CIG_CEILING_QUALITY_CODE', CIG.getItem(1).cast('int')) # categorical
        .withColumn('CIG_CEILING_DETERMINATION_CODE', CIG.getItem(2).cast('string')) # categorical 
        .withColumn('CIG_CAVOK_CODE', CIG.getItem(3).cast('string')) # categorical/binary
        
        # VIS Fields
        .withColumn('VIS_DISTANCE_DIMENSION', VIS.getItem(0).cast('int')) # numeric
        .withColumn('VIS_DISTANCE_QUALITY_CODE', VIS.getItem(1).cast('int')) # categorical
        .withColumn('VIS_VARIABILITY_CODE', VIS.getItem(2).cast('string')) # categorical/binary
        .withColumn('VIS_QUALITY_VARIABILITY_CODE', VIS.getItem(3).cast('int')) # categorical
        
        # TMP
        .withColumn('TMP_AIR_TEMP', TMP.getItem(0).cast('int')) # numeric
        .withColumn('TMP_AIR_TEMP_QUALITY_CODE', TMP.getItem(1).cast('string')) # categorical
        
        # DEW
        .withColumn('DEW_POINT_TEMP', DEW.getItem(0).cast('int')) # numeric
        .withColumn('DEW_POINT_QUALITY_CODE', DEW.getItem(1).cast('string')) # categorical
        
        # SLP
        .withColumn('SLP_SEA_LEVEL_PRES', SLP.getItem(0).cast('int')) # numeric
        .withColumn('SLP_SEA_LEVEL_PRES_QUALITY_CODE', SLP.getItem(1).cast('int')) # categorical
       ).drop('WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP')
  
  return df

df_weather_subset_decoded = weather_decode(df_weather_h)

# COMMAND ----------

df_weather_subset_decoded.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Treat Erroneous Data

# COMMAND ----------

# Remove invalid data as per documentation

# Keep records with valid continuous measurements
continuous_col_list = ['WND_DIRECTION_ANGLE',
                      'WND_SPEED_RATE',
                      'CIG_CEILING_HEIGHT_DIMENSION',
                      'VIS_DISTANCE_DIMENSION',
                      'TMP_AIR_TEMP',
                      'DEW_POINT_TEMP',
                      'SLP_SEA_LEVEL_PRES']

# Cast to integer
for col in continuous_col_list:
  df_weather = df_weather_subset_decoded.withColumn(col, f.col(col).cast('int'))

df_weather = df_weather_subset_decoded.filter(
((f.col('WND_DIRECTION_ANGLE') >= 1) & (f.col('WND_DIRECTION_ANGLE') <= 360)) | (f.col('WND_DIRECTION_ANGLE') == 999))\
.filter(((f.col('WND_SPEED_RATE') >= 0) & (f.col('WND_SPEED_RATE') <= 900)) | (f.col('WND_SPEED_RATE') == 9999))\
.filter(((f.col('CIG_CEILING_HEIGHT_DIMENSION') >= 0) & (f.col('CIG_CEILING_HEIGHT_DIMENSION') <= 22000)) | (f.col('CIG_CEILING_HEIGHT_DIMENSION') == 99999))\
.filter(((f.col('VIS_DISTANCE_DIMENSION') >= 0) & (f.col('VIS_DISTANCE_DIMENSION') <= 160000)) | (f.col('VIS_DISTANCE_DIMENSION') == 999999))\
.filter(((f.col('TMP_AIR_TEMP') >= -932) & (f.col('TMP_AIR_TEMP') <= 618)) | (f.col('TMP_AIR_TEMP') == 9999))\
.filter(((f.col('DEW_POINT_TEMP') >= -982) & (f.col('DEW_POINT_TEMP') <= 368)) | (f.col('DEW_POINT_TEMP') == 9999))\
.filter(((f.col('SLP_SEA_LEVEL_PRES') >= 8600) & (f.col('SLP_SEA_LEVEL_PRES') <= 10900)) | (f.col('SLP_SEA_LEVEL_PRES') == 99999))
  
# Keep records with valid categorical features
categorical_col_list = ['WND_QUALITY_CODE', 
                        'WND_SPEED_QUALITY_CODE',
                        'WND_TYPE_CODE',
                        'CIG_CEILING_QUALITY_CODE',
                        'CIG_CEILING_DETERMINATION_CODE',
                        'CIG_CAVOK_CODE',
                        'VIS_DISTANCE_QUALITY_CODE',
                        'VIS_QUALITY_VARIABILITY_CODE',
                        'VIS_VARIABILITY_CODE',
                        'TMP_AIR_TEMP_QUALITY_CODE',
                        'DEW_POINT_QUALITY_CODE',
                        'SLP_SEA_LEVEL_PRES_QUALITY_CODE']

# Cast to string
for col in categorical_col_list:
  df_weather = df_weather.withColumn(col, f.col(col).cast('string'))
  
df_weather = df_weather.filter(f.col('WND_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('WND_SPEED_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('CIG_CEILING_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('VIS_DISTANCE_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('VIS_QUALITY_VARIABILITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('SLP_SEA_LEVEL_PRES_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9"}))\
                       .filter(f.col('TMP_AIR_TEMP_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9", "A", "C", "I", "M", "P", "R", "U"}))\
                       .filter(f.col('DEW_POINT_QUALITY_CODE').isin({"0", "1", "2", "4", "5", "6", "9", "A", "C", "I", "M", "P", "R", "U"}))\
                       .filter(f.col('WND_TYPE_CODE').isin({"A", "B", "C", "H", "N", "R", "Q", "T", "V", "9"}))\
                       .filter(f.col('CIG_CEILING_DETERMINATION_CODE').isin({"A", "B", "C", "D", "E", "M", "P", "R", "S", "U", "V", "W", "9"}))\
                       .filter(f.col('CIG_CAVOK_CODE').isin({"N", "Y", "9"}))\
                       .filter(f.col('VIS_VARIABILITY_CODE').isin({"N", "V", "9"}))\
                       .drop('WND_QUALITY_CODE', 'WND_SPEED_QUALITY_CODE', 'CIG_CEILING_QUALITY_CODE',
                             'VIS_DISTANCE_QUALITY_CODE', 'VIS_QUALITY_VARIABILITY_CODE',
                             'TMP_AIR_TEMP_QUALITY_CODE','SLP_SEA_LEVEL_PRES_QUALITY_CODE', 
                             'DEW_POINT_QUALITY_CODE')

# df_weather.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute Null Weather data
# MAGIC Now that we have replaced the "9's" codes with Nulls, imputation will be a bit easier. We choose a 7 day lookback average to impute null data

# COMMAND ----------

# Label column values with error free weather data
df_weather_valid = df_weather.withColumn('VALID_WEATHER_DATA', 
                                         f.when((f.col('WND_DIRECTION_ANGLE') == 999) | 
                                                (f.col('WND_SPEED_RATE') == 9999) | 
                                                (f.col('CIG_CEILING_HEIGHT_DIMENSION') == 99999) | 
                                                (f.col('VIS_DISTANCE_DIMENSION') == 999999) | 
                                                (f.col('TMP_AIR_TEMP') == 9999) | 
                                                (f.col('DEW_POINT_TEMP') == 9999) | 
                                                (f.col('SLP_SEA_LEVEL_PRES') == 99999), 0).otherwise(1))

# Replace 9s with null values
df_weather_valid_fillna =  df_weather_valid.withColumn('WND_DIRECTION_ANGLE',
                                           f.when(df_weather_valid.WND_DIRECTION_ANGLE == 999, None)\
                                           .otherwise(df_weather_valid.WND_DIRECTION_ANGLE))\
                                           .withColumn('WND_SPEED_RATE',
                                           f.when(df_weather_valid.WND_SPEED_RATE == 9999, None)\
                                           .otherwise(df_weather_valid.WND_SPEED_RATE))\
                                           .withColumn('CIG_CEILING_HEIGHT_DIMENSION',
                                           f.when(df_weather_valid.CIG_CEILING_HEIGHT_DIMENSION == 99999, None)\
                                           .otherwise(df_weather_valid.CIG_CEILING_HEIGHT_DIMENSION))\
                                           .withColumn('VIS_DISTANCE_DIMENSION',
                                           f.when(df_weather_valid.VIS_DISTANCE_DIMENSION == 999999, None)\
                                           .otherwise(df_weather_valid.VIS_DISTANCE_DIMENSION))\
                                           .withColumn('TMP_AIR_TEMP',
                                           f.when(df_weather_valid.TMP_AIR_TEMP == 9999, None)\
                                           .otherwise(df_weather_valid.TMP_AIR_TEMP))\
                                           .withColumn('DEW_POINT_TEMP',
                                           f.when(df_weather_valid.DEW_POINT_TEMP == 9999, None)\
                                           .otherwise(df_weather_valid.DEW_POINT_TEMP))\
                                           .withColumn('SLP_SEA_LEVEL_PRES',
                                           f.when(df_weather_valid.SLP_SEA_LEVEL_PRES == 99999, None)\
                                           .otherwise(df_weather_valid.SLP_SEA_LEVEL_PRES))\
                                           .withColumn('WND_TYPE_CODE',
                                           f.when(df_weather_valid.WND_TYPE_CODE == 9, None)\
                                           .otherwise(df_weather_valid.WND_TYPE_CODE))\
                                           .withColumn('CIG_CAVOK_CODE',
                                           f.when(df_weather_valid.CIG_CAVOK_CODE == 9, None)\
                                           .otherwise(df_weather_valid.CIG_CAVOK_CODE))\
                                           .withColumn('VIS_VARIABILITY_CODE',
                                           f.when(df_weather_valid.VIS_VARIABILITY_CODE == 9, None)\
                                           .otherwise(df_weather_valid.VIS_VARIABILITY_CODE))

# Impute null values
seven_day_window = Window.partitionBy('STATION').orderBy(f.col('DATE').cast('long')).rangeBetween(-7*86400, 0)

col_list = [
  'WND_DIRECTION_ANGLE',
  'WND_SPEED_RATE',
  'CIG_CEILING_HEIGHT_DIMENSION',
  'VIS_DISTANCE_DIMENSION',
  'TMP_AIR_TEMP',
  'DEW_POINT_TEMP',
  'SLP_SEA_LEVEL_PRES',
]

impute_col_list = [
  'WND_DIRECTION_ANGLE-AVG',
  'WND_SPEED_RATE-AVG',
  'CIG_CEILING_HEIGHT_DIMENSION-AVG',
  'VIS_DISTANCE_DIMENSION-AVG',
  'TMP_AIR_TEMP-AVG',
  'DEW_POINT_TEMP-AVG',
  'SLP_SEA_LEVEL_PRES-AVG',
]

for col, imp_col in zip(col_list, impute_col_list):
  df_weather_valid_fillna = df_weather_valid_fillna.withColumn(imp_col, f.when(f.col(col).isNull(),
                                                              f.avg(col).over(seven_day_window))\
                                                              .otherwise(f.col(col)))


# COMMAND ----------

# Save data to Blob
WEATHER_FULL_PROCESSED_PATH = blob_url + '/processed/weather_full.parquet'
df_weather_valid_fillna.mode('overwrite').write.parquet(WEATHER_FULL_PROCESSED_PATH)

# COMMAND ----------

# Read data from Blob
# weather_sample_processed = spark.read.parquet(f'{blob_url}/30k_weather_sample_processed').cache()
weather_temp = spark.read.parquet(f'{WEATHER_FULL_PROCESSED_PATH}').cache()
# sc = spark.sparkContext
# e = sc.broadcast(weather_sample_processed)

# COMMAND ----------

weather_temp.count()

# COMMAND ----------

type(weather_sample_processed)

# COMMAND ----------

