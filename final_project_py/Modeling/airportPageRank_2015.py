# Databricks notebook source
# MAGIC %md
# MAGIC # Airport Page Rank

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Data

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# set main_data
main_data = airline_2015_processed_df.filter(f.col('MONTH') < 8).cache()

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

spark_df = airport_rank.vertices.orderBy("pagerank", ascending=False)
joined_full_null_imputed_df_pagerank = joined_full_null_imputed_df.join(
    spark_df.select('id', 'pagerank'), 
    joined_full_null_imputed_df.ORIGIN == spark_df.id,
    "left").drop('id').cache()
joined_full_null_imputed_df_pagerank = joined_full_null_imputed_df_pagerank.withColumn(
    'pagerank', f.when(f.col('pagerank').isNull(), 0).otherwise(f.col('pagerank'))
)

JOINED_FULL_NULL_IMPUTED_PAGERANK_PROCESSED_PATH = blob_url + '/processed/joined_full_null_imputed_processed_pagerank_df.parquet'
 
joined_full_null_imputed_df_pagerank.write.mode('overwrite').parquet(JOINED_FULL_NULL_IMPUTED_PAGERANK_PROCESSED_PATH)

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
        title = 'Y2015 M1-7 Airport PageRank',
        geo = dict(projection_type ='albers usa'),
    )
fig.show()

# COMMAND ----------

