# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import * # StringType
from pyspark.sql import Window
import pandas as pd
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

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Loading weather data before July 1st, 2015

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-07-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC **Columns in the weather data dataframe**

# COMMAND ----------

sorted(df_weather.columns)

# COMMAND ----------

df_weather.printSchema()

# COMMAND ----------

df_weather.dtypes

# COMMAND ----------

(df_weather.count(), len(df_weather.columns))

# COMMAND ----------

display(df_weather)

# COMMAND ----------

display(df_weather.select([(count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           ))/df_weather.count()).alias(c)
                    for c in df_weather.columns]))

# COMMAND ----------

import pandas as pd

df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/abajaj225@berkeley.edu/weather_6months_null_dataset_percentage.csv")
df1 = df1.T

# COMMAND ----------

#Saved the results of the above in weather_6months_null_dataset_percentage.csv
# The above cell to check for nulls took 5.87 hours to run.
import pandas as pd

df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/abajaj225@berkeley.edu/weather_6months_null_dataset_percentage.csv")
df1 = df1.T

# COMMAND ----------

df1=df1.reset_index(drop=False).rename(columns={'index':'col_name',0:'null_percent'})

# COMMAND ----------

df1=df1.reset_index(drop=False).rename(columns={'index':'col_name',0:'null_percent'})
weather_6mon_ninety_percent_null_values = df1[df1['null_percent']>=0.9]['col_name'].values
weather_6mon_seventy_percent_null_values = df1[(df1['null_percent']>=0.7) & (df1['null_percent']<0.9) ]['col_name'].values
weather_6mon_seventy_percent_null_values

# COMMAND ----------

weather_6mon_ninety_percent_null_values = df1[df1['null_percent']>=0.9]['col_name'].values
weather_6mon_ninety_percent_null_values

# COMMAND ----------

df_weather = df_weather.drop(*weather_6mon_ninety_percent_null_values)
df_weather = df_weather.drop(*weather_6mon_seventy_percent_null_values)

# COMMAND ----------

weather_6mon_seventy_percent_null_values = df1[(df1['null_percent']>=0.7) & (df1['null_percent']<0.9) ]['col_name'].values
weather_6mon_seventy_percent_null_values

# COMMAND ----------

df_weather = df_weather.drop(*weather_6mon_ninety_percent_null_values)
df_weather = df_weather.drop(*weather_6mon_seventy_percent_null_values)

# COMMAND ----------

df_weather.count(),len(df_weather.columns)

# COMMAND ----------

display(df_weather)

# COMMAND ----------

# df_weather = df_weather.withColumn("DATE", df_weather["DATE"].cast(StringType()))
# df_weather = df_weather.withColumn("STATION", df_weather["STATION"].cast(IntegerType()))

# COMMAND ----------

# display(df_weather)#.select(df_weather['STATION']))
df_weather.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Loading stations data

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*").cache()
display(df_stations)

# COMMAND ----------

# Size of the dataframe

print(f"Size of the stations df: {df_stations.count(),len(df_stations.columns)}")
df_stations.printSchema()

# COMMAND ----------

# Count of distinct values in the dataframe
for col in df_stations.columns:
  print(f"Col {col} ,{df_stations.select(col).distinct().count()}")

# COMMAND ----------

# Adding monotonically increasing ID to use as a key for a join
df_stations = df_stations.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

['usaf',
 'wban',
 'station_id',
 'lat',
 'lon',
 'neighbor_id',
 'neighbor_name',
 'neighbor_state',
 'neighbor_call',
 'neighbor_lat',
 'neighbor_lon',
 'distance_to_neighbor']

# COMMAND ----------

df_limited = df_stations.sort('id').limit(50)

# COMMAND ----------

df_limited.display()

# COMMAND ----------

df_limited.groupBy('neighbor_call').agg(collect_list("distance_to_neighbor"),collect_list("station_id")).display()

# COMMAND ----------

aggregation_window = Window.partitionBy('neighbor_call')
df_stations.select([col for col in df_limited.columns]).withColumn('min_distance_to_neighbor', min('distance_to_neighbor').over(aggregation_window)).display()

# COMMAND ----------

def get_neighbours_stations(stations,distance):
  