# Databricks notebook source
# MAGIC %md
# MAGIC # Station Data Processing
# MAGIC 
# MAGIC The purpose of this notebook is to organize all of the data processing actions that were taken throughout the EDA flow. The final output is a parquet file that will be uploaded to the blob for usage in transformations

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
df_stations = spark.read.parquet('/mnt/mids-w261/datasets_final_project/stations_data/*').cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a table that maps the closest station to ICAO code
# MAGIC Station is found in `WEATHER`, and we have joined `ICAO` into `AIRLINES` so we can now use this table to find the closest weather station

# COMMAND ----------

# Select Relevant Columns only
df_stations = df_stations.select(['neighbor_call', 'station_id', 'distance_to_neighbor']).orderBy(['neighbor_call','distance_to_neighbor'])


# COMMAND ----------

# For each ICAO (airport), find the closest weather station
station_to_ICAO = df_stations.withColumn('row_number', f.row_number()\
                                 .over(Window.partitionBy(f.col('neighbor_call'))\
                                 .orderBy(f.col('distance_to_neighbor').asc())))\
                                 .where(f.col('row_number') == 1)\
                                 .drop(f.col('row_number'))

# COMMAND ----------

# Rename some columns
ICAO_station_mapping = station_to_ICAO.withColumnRenamed('neighbor_call', 'ICAO')\
                                      .withColumnRenamed('station_id', 'WEATHER_STATION_ID')\
                                      .drop('distance_to_neighbor') # After inspection, can see the distance is 0 for all included stations - so weather station is on airport grounds

# COMMAND ----------

# Save Data to BLOB
ICAO_station_mapping.write.mode("overwrite").parquet(f'{blob_url}/ICAO_station_mapping')

# COMMAND ----------

# Read data from Blob
ICAO_station_mapping = spark.read.parquet(f'{blob_url}/ICAO_station_mapping').cache()

# COMMAND ----------

ICAO_station_mapping.count()

# COMMAND ----------

ICAO_station_mapping.select('ICAO_DIST_TO_STATION').distinct().display()

# COMMAND ----------

df_stations = spark.read.parquet('/mnt/mids-w261/datasets_final_project/stations_data/*').cache()
df_stations.count()

# COMMAND ----------

df_stations.display()

# COMMAND ----------

