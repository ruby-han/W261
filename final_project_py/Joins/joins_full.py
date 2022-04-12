# Databricks notebook source
# MAGIC %md
# MAGIC #### Load Data

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# # Load data from Saved Parquet Files | 

# ICAO_station_mapping = spark.read.parquet(f'{blob_url}/ICAO_station_mapping').cache()

# Quick rename for sake of brevity
flights = airline_full_processed_df
stations = ICAO_station_mapping
weather = weather_full_processed_df

# COMMAND ----------

display(weather_full_processed_df)

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

origin_weather.printSchema()

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

# display(flights_w_weather)

# COMMAND ----------

JOINED_FULL_PROCESSED_PATH = blob_url + '/processed/joined_full_processed_df.parquet'

flights_w_weather.write.parquet(JOINED_FULL_PROCESSED_PATH)

# COMMAND ----------


flights_in = spark.read.parquet(JOINED_FULL_PROCESSED_PATH).cache()

# COMMAND ----------

flights_in.count()

# COMMAND ----------

# flights_in.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()).groupBy(f.col('OP_UNIQUE_CARRIER')).count().display()
flights_in.where((f.col('ORIGIN_WEATHER_STATION').isNull()) | (f.col('DEST_WEATHER_STATION').isNull())).count()