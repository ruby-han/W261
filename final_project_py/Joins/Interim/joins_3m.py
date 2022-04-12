# Databricks notebook source
# Load Packages, set storage links |

from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql.functions import col, to_timestamp, to_utc_timestamp, concat_ws, udf
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import airporttime

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

# MAGIC %md
# MAGIC #### Load Data

# COMMAND ----------

# Load data from Saved Parquet Files | 

# flights_3m = spark.read.parquet(f'{blob_url}/3m_flights_processed').cache()
# flights_6m = spark.read.parquet(f'{blob_url}/6m_flights_processed').cache()
ICAO_station_mapping = spark.read.parquet(f'{blob_url}/ICAO_station_mapping').cache()
# weather_sample_processed = spark.read.parquet(f'{blob_url}/30k_weather_sample_processed').cache()
# weather_sample_processed_new = spark.read.parquet(f'{blob_url}/weather_2015_02_21').cache()

# 3 months of airline data
AIRLINE_3M_PROCESSED_PATH = blob_url + '/processed/airline_3m_data.parquet'
flight_data = spark.read.parquet(AIRLINE_3M_PROCESSED_PATH).cache()

# 3 months of weather data
WEATHER_3M_PROCESSED_PATH = blob_url + '/weather_2015_Q1'
weather_data = spark.read.parquet(WEATHER_3M_PROCESSED_PATH).cache()

# Quick rename for sake of brevity
flights = flight_data
stations = ICAO_station_mapping
weather = weather_data

# COMMAND ----------

display(weather_sample_processed)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Schema Checks

# COMMAND ----------

# Schema Checks | 

# flights.printSchema()
# stations.printSchema()
# weather.printSchema()

# COMMAND ----------

# First, join flights to stations on ICAO - airport code which is a unique identifier we can use for both origin and destination
# In this join, we will be including:
#    - `WEATHER_STATION_ID`: the closest weather station which will then allow us to pull in weather at each airport

flights_w_stations = flights.join(stations, flights.ICAO_ORIGIN == stations.ICAO, 'left')\
                            .select(flights['*'], stations['WEATHER_STATION_ID'].alias('ORIGIN_WEATHER_STATION_ID'))\
                            .join(stations, flights.ICAO_DEST == stations.ICAO, 'left')\
                            .select(flights['*'], 'ORIGIN_WEATHER_STATION_ID', stations['WEATHER_STATION_ID'].alias('DEST_WEATHER_STATION_ID'))

# COMMAND ----------

# Sanity Check
# flights_w_stations.select(['ORIGIN', 'DEST', 'ICAO_ORIGIN', 'ICAO_DEST', 'ORIGIN_WEATHER_STATION_ID', 'DEST_WEATHER_STATION_ID']).distinct().where(f.col('ICAO_DEST').isin(['KORD', 'KATL', 'KFSD', 'KROC', 'KTPA', 'KGPI', 'KOAK', 'KRAP'])).display()
# stations.where(f.col('ICAO').isin(['KORD', 'KATL', 'KFSD', 'KROC', 'KTPA', 'KGPI', 'KOAK', 'KRAP'])).display()
# flights_w_stations.printSchema()

# COMMAND ----------

# Check to make sure new timestamp mapping looks correct
# flights_w_stations.select('CRS_DEP_TIME_UTC_HOUR').withColumn('new_time', to_timestamp(f.col('CRS_DEP_TIME_UTC_HOUR').cast('long') - 10800)).display()

# COMMAND ----------

# Second, we need to create a new timestamp to join weather on - which is lagged by 2 hours. 
# Due to the nature of our timestamps - we will actually have to subtract 3 hours to avoid leakage. 10800 is equivalent to 3 hours
flights_w_stations = flights_w_stations.withColumn('CRS_DEP_TIME_UTC_LAG', to_timestamp(f.col('CRS_DEP_TIME_UTC_HOUR').cast('long') - 10800))

# Next, prepend an origin or destination prefix to the weather columns, so when we join we know which weather set we're looking at
origin_weather = weather.select([f.col(weather_feat).alias('ORIGIN_WEATHER_'+weather_feat) for weather_feat in weather.columns])
dest_weather = weather.select([f.col(weather_feat).alias('DEST_WEATHER_'+weather_feat) for weather_feat in weather.columns])

# join flights to ORIGIN weather on station_id and our lagged ORIGIN time variable
flights_w_weather_temp = flights_w_stations.join(origin_weather, (flights_w_stations.ORIGIN_WEATHER_STATION_ID == origin_weather.ORIGIN_WEATHER_STATION) &\
                                                          (flights_w_stations.CRS_DEP_TIME_UTC_LAG == origin_weather.ORIGIN_WEATHER_HOUR), 
                                                          'left')

# Finally, join flights to DESTINATION weather on station_id and lagged ORIGIN time variable
flights_w_weather = flights_w_weather_temp.join(dest_weather, (flights_w_weather_temp.DEST_WEATHER_STATION_ID == dest_weather.DEST_WEATHER_STATION) &\
                                                          (flights_w_weather_temp.CRS_DEP_TIME_UTC_LAG == dest_weather.DEST_WEATHER_HOUR), 
                                                          'left')

# flights_w_weather_temp.count()
# flights_w_weather_temp.where(f.col('ID') == '2015-02-01N3MEAAORD730').display()

# COMMAND ----------

display(flights_w_weather)

# COMMAND ----------

# flights_w_weather_temp.where(f.col('FL_DATE') == '2015-02-21').display()
# flights_w_weather_temp.where(f.col('ID') == '2015-02-21N4XDAAORD1805').display()
# flights_w_weather.where(f.col('ID') == '2015-02-21N4XDAAORD1805').display()

# COMMAND ----------

JOINED_3M_PROCESSED_PATH = blob_url + '/processed/joined_3m_data.parquet'

flights_w_weather.write.parquet(JOINED_3M_PROCESSED_PATH)

# COMMAND ----------


flights_in = spark.read.parquet(JOINED_3M_PROCESSED_PATH).cache()

# COMMAND ----------

# Null values
def null_values(df):
  '''Pass pandas df argument and return columns with null values and percentages.'''
  df = df.replace('', np.nan)
  null_values = df.isna().sum()
  null_values_percent = null_values/len(df) * 100
  null_table = pd.concat([null_values, null_values_percent], axis=1).rename(
    columns = {0:'Null Values', 1:'Percentage'})
  null_table = null_table[null_table.Percentage != 0]
  sorted_table = null_table.sort_values('Percentage', ascending=False)
  
  print(f'''Total Number of Columns: {df.shape[1]}\nNumber of Columns with Null Values: {sorted_table.shape[0]}''')
  
  return sorted_table

# COMMAND ----------

flights_in.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()).groupBy(f.col('OP_UNIQUE_CARRIER')).count().display()

# COMMAND ----------

flights_in.printSchema()

# COMMAND ----------

df = flights_in.toPandas()
null_df = null_values(df)

null_df

# COMMAND ----------

flights_in.display()

# COMMAND ----------

weather.where(((f.col('DATE') > "2015-02-20T23:59:59.000") & (f.col('DATE') < "2015-02-22T00:00:00.000")) & (f.col('STATION') == '72211012842')).display() 

# COMMAND ----------

weather.printSchema()

# COMMAND ----------

stations2 = stations
prefix = 'ORIGIN'
stations3 = stations2.select([col(c).alias(prefix+c) for c in stations2.columns])
stations3.display()

# COMMAND ----------

flights_w_weather_temp2 = flights_w_stations.join(weather, (flights_w_stations.ORIGIN_WEATHER_STATION_ID == weather.STATION) &\
                                                          (flights_w_stations.CRS_DEP_TIME_UTC_HOUR == weather.HOUR))

flights_w_weather_temp2.display()

# COMMAND ----------

weather.filter((col('DATE') > "2015-02-20T23:59:59.000") & (col('DATE') < "2015-02-22T00:00:00.000")).display()

# COMMAND ----------

flights.where(f.col('CRS_DEP_TIME_UTC_HOUR') == '2015-02-26T13:00:00.000+0000').distinct().display()

# COMMAND ----------

flights_in = spark.read.parquet(JOINED_3M_PROCESSED_PATH).cache()