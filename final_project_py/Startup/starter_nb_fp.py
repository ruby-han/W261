# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

