# Databricks notebook source
# MAGIC %md
# MAGIC # Airport Page Rank

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Data

# COMMAND ----------

# load packages
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import datetime

from pyspark.sql import functions as f
from pyspark.sql.functions import col, to_timestamp, to_utc_timestamp, concat_ws, udf
from pyspark.sql import Window

from graphframes import *

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

JOINED_3M_PROCESSED_PATH = blob_url + '/processed/joined_3m_data.parquet'
JOINED_2015_PROCESSED_PATH = blob_url + '/processed/joined_2015_data.parquet'

# read raw data
airline_full_raw_df = spark.read.parquet(AIRLINE_FULL_PATH).cache()
airline_3m_raw_df = spark.read.parquet(AIRLINE_3M_PATH).cache()
airline_6m_raw_df = spark.read.parquet(AIRLINE_6M_PATH).cache()

# read processed data
airline_full_processed_df = spark.read.parquet(AIRLINE_FULL_PROCESSED_PATH).cache()
airline_2015_processed_df = spark.read.parquet(AIRLINE_2015_PROCESSED_PATH).cache()
airline_2019_processed_df = spark.read.parquet(AIRLINE_2019_PROCESSED_PATH).cache()

joined_3m_processed_df = spark.read.parquet(JOINED_3M_PROCESSED_PATH).cache()
joined_2015_processed_df = spark.read.parquet(JOINED_2015_PROCESSED_PATH).cache()

# set main_data
main_data = airline_full_processed_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Investigate Blob Azure Storage

# COMMAND ----------

display(dbutils.fs.ls(blob_url))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Size

# COMMAND ----------

def file_ls(path: str):
    '''List all files in base path recursively.'''
    tot = 0
    for x in dbutils.fs.ls(path):
        if x.path[-1] != '/':
            tot += x.size
            yield x
        else:
            for y in file_ls(x.path):
                yield y
    yield f'DATASIZE: {tot}'

total_size = []
for i in file_ls(AIRLINE_2019_PROCESSED_PATH):
    if 'DATASIZE:' in i:
        total_size.append(int(i.split(' ')[1]))

print(f'Total Data Size: {sum(total_size)/1e9:.2f} GB')
# print(f'Total Number of Records: {main_data.count():,}')

# COMMAND ----------

main_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create GraphFrame + PageRank

# COMMAND ----------

ORIGIN = main_data.select('ORIGIN', 'ORIGIN_CITY_NAME', 'AIRPORT_LAT_ORIGIN', 'AIRPORT_LONG_ORIGIN').distinct()
DEST = main_data.select('DEST', 'DEST_CITY_NAME', 'AIRPORT_LAT_DEST', 'AIRPORT_LONG_DEST').distinct()

AIRPORT = ORIGIN.union(DEST).distinct()
AIRPORT = AIRPORT.withColumnRenamed('ORIGIN', 'id')\
                 .withColumnRenamed('ORIGIN_CITY_NAME', 'name')

AIRPORT_EDGES = (
    main_data.select(
        f.col('ORIGIN').alias('src'),
        f.col('DEST').alias('dst'),
        'OP_UNIQUE_CARRIER','MONTH','QUARTER','YEAR','ORIGIN_CITY_NAME','DEST_CITY_NAME','DISTANCE',
        'AIRPORT_LAT_ORIGIN', 'AIRPORT_LONG_ORIGIN', 'AIRPORT_LAT_DEST', 'AIRPORT_LONG_DEST',
        f.format_string('%d-%02d',f.col('YEAR'),f.col('MONTH')).alias('YEAR-MONTH')        
    )
).cache()

airport_graph = GraphFrame(AIRPORT, AIRPORT_EDGES)
airport_rank = airport_graph.pageRank(resetProbability=0.15, maxIter=5).cache()

# COMMAND ----------

# top 20 busiest airport hubs
airport_rank_pd_df = airport_rank.vertices.orderBy("pagerank", ascending=False).toPandas()
(airport_rank.vertices.orderBy("pagerank", ascending=False)).limit(20).toPandas()

# COMMAND ----------

# normalize page rank
airport_rank_pd_df['norm_pageRank'] = airport_rank_pd_df['pagerank']/airport_rank_pd_df['pagerank'].max()
airport_rank_pd_df['norm_pageRank'] = airport_rank_pd_df['norm_pageRank'].round(2)

# label
airport_rank_pd_df['label'] = 'IATA: ' + airport_rank_pd_df['id'] + ', PageRank: ' + airport_rank_pd_df['norm_pageRank'].astype('str')

# plot geomap
fig = go.Figure(
    data=go.Scattergeo(
        
        locationmode = 'USA-states',
        lat = airport_rank_pd_df['AIRPORT_LAT_ORIGIN'],
        lon = airport_rank_pd_df['AIRPORT_LONG_ORIGIN'],
        text = airport_rank_pd_df['label'],
        mode = 'markers',
        marker = dict(size = airport_rank_pd_df['pagerank']*2,
                      color = airport_rank_pd_df['pagerank'],
                      colorbar_title = 'Rank')
    )
)

fig.update_layout(
        title = 'Y2015-2019 Airport PageRank',
        geo = dict(projection_type ='albers usa'),
    )
fig.show()

# COMMAND ----------

