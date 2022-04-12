# Databricks notebook source
# MAGIC %md
# MAGIC # Airline Data Processing
# MAGIC 
# MAGIC The purpose of this notebook is to organize all of the data processing actions that were taken throughout the EDA flow. The final output is a parquet file that will be uploaded to the blob for usage in transformations

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Data
# MAGIC Team note - no real changes here yet from starter_nb - minimizing

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

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
for i in file_ls(PARQ_AIRLINE_PATH):
    if 'DATASIZE:' in i:
        total_size.append(int(i.split(' ')[1]))

print(f'Total Data Size: {sum(total_size)/1e9:.2f} GB')
# print(f'Total Number of Records: {main_data.count():,}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Keep Selected Features
# MAGIC - Excluded features with 90%+ null proportions as discussed in EDA notebook
# MAGIC 
# MAGIC #### 2. Remove Duplicate Data
# MAGIC - Created a unique ID by combining the below:
# MAGIC   - `ID` - `FL_DATE`, `CRS_DEP_TIME`, `ORIGIN`, `TAIL_NUM`
# MAGIC - Upon unique ID creation, discovered duplicate flights
# MAGIC   - Some flights were cancelled and replaced by flights to a different destination
# MAGIC     - Remove cancelled flights and keep replacement flights
# MAGIC   - Some flights have same tail_num and origin but different destination
# MAGIC     - Remove the above as data seems suspicious
# MAGIC     
# MAGIC #### 3. Remove Cancelled and Diverted Flights
# MAGIC - Discussed in EDA notebook
# MAGIC 
# MAGIC #### 4. Remove Null Time Entries

# COMMAND ----------

# keep selected features
FEATURES = eval(dbutils.notebook.run('/Users/rubyhan@berkeley.edu/team28/Final_Project/featureSelection', 60))
main_data_df = main_data.select(FEATURES['AIRLINE']).cache()

# add unique ID (FL_DATE-CRS_DEP_TIME-ORIGIN-TAIL_NUM)
main_data_df = main_data_df.withColumn('ID'
                                       ,f.concat(
                                           f.col('FL_DATE')
                                           ,f.lit('-')
                                           ,f.col('CRS_DEP_TIME')
                                           ,f.lit('-')
                                           ,f.col('ORIGIN')
                                           ,f.lit('-')
                                           ,f.col('TAIL_NUM')
                                       )).where(f.col('ID').isNotNull())

# identify duplicates based on new 'ID' column
main_data_df_unique_count = main_data_df.groupBy('ID').count()
duplicate_ID = main_data_df_unique_count.filter(f.col('count') > 1).cache()

# print(f'Duplicate Counts of ID: {duplicate_ID.count():,}')

# drop duplicate IDs
dup_list = [x['ID'] for x in duplicate_ID.select('ID').collect()]
main_data_df = main_data_df.filter(main_data_df['ID'].isin(dup_list) == False).cache()

# remove cancelled and diverted flights (not the focus of project - delayed flights are)
main_data_df = main_data_df.filter(
    (f.col('CANCELLED') == 0) & (f.col('DIVERTED') == 0)
).cache()

# remove null time entries
main_data_df = main_data_df.where(f.col('DEP_TIME').isNotNull() &
                                  f.col('ARR_TIME').isNotNull() & 
                                  f.col('CRS_DEP_TIME').isNotNull() &
                                  f.col('CRS_ARR_TIME').isNotNull()).cache()

# print(f'Number of Records: {main_data_df.count():,}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Convert to UTC Timestamps
# MAGIC - Weather data is in UTC and airline data is in local time 
# MAGIC - Convert to have relevant link between two datasets

# COMMAND ----------

# pull in airport coordinates, ICAO, other features for join to convert to UTC
airport_coord = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)

# rename headers
airport_coord = airport_coord.rename(columns=
                                       {0: 'AIRPORT_ID',
                                        1: 'AIRPORT_NAME',
                                        2: 'AIRPORT_CITY',
                                        3: 'AIRPORT_COUNTRY',
                                        4: 'IATA',
                                        5: 'ICAO',
                                        6: 'AIRPORT_LAT',
                                        7: 'AIRPORT_LONG',
                                        8: 'AIRPORT_ALT',
                                        9: 'AIRPORT_UTC_OFFSET',
                                        10: 'AIRPORT_DST',
                                        11: 'AIRPORT_TIMEZONE',
                                        12: 'AIRPORT_TYPE',
                                        13: 'AIRPORT_DATASOURCE'})

# convert to spark df
airport_coord = spark.createDataFrame(airport_coord)

# join on airport_coord to airlines data based on origin
KEY_list = ['IATA', 'ICAO', 'AIRPORT_LAT', 'AIRPORT_LONG', 'AIRPORT_TIMEZONE', 'AIRPORT_UTC_OFFSET']
airport_coord_origin = main_data_df.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_ORIGIN'),
                                                            'ICAO', f.col('ICAO').alias('ICAO_ORIGIN'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_ORIGIN'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_ORIGIN'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_ORIGIN'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_ORIGIN')),
                                                            airport_coord.IATA == main_data_df.ORIGIN).drop(*KEY_list)
flight_data = airport_coord_origin.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_DEST'),
                                                             'ICAO', f.col('ICAO').alias('ICAO_DEST'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_DEST'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_DEST'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_DEST'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_DEST')),
                                                            airport_coord.IATA == main_data_df.DEST).drop(*KEY_list)

time_list = [flight_data.CRS_DEP_TIME, flight_data.DEP_TIME, flight_data.ARR_TIME, flight_data.CRS_ARR_TIME]
utc_time_list = ['CRS_DEP_TIME_UTC', 'DEP_TIME_UTC', 'ARR_TIME_UTC', 'CRS_ARR_TIME_UTC']

for time_col, utc_time_col in zip(time_list[:2], utc_time_list[:2]):
    flight_data = flight_data.withColumn(utc_time_col, 
                       f.to_timestamp(
                         f.concat(
                           f.col('FL_DATE'),
                           f.lpad(time_col, 4, '0')
                         ), format='yyyy-MM-ddHHmm'
                       ))\
            .withColumn(utc_time_col,
                       f.to_utc_timestamp(f.col(utc_time_col), 
                                          f.col('AIRPORT_TIMEZONE_ORIGIN'))
                       )

for time_col, utc_time_col in zip(time_list[-2:], utc_time_list[-2:]):
    flight_data = flight_data.withColumn(utc_time_col, 
                       f.to_timestamp(
                         f.concat(
                           f.col('FL_DATE'),
                           f.lpad(time_col, 4, '0')
                         ), format='yyyy-MM-ddHHmm'
                       ))\
            .withColumn(utc_time_col,
                       f.to_utc_timestamp(f.col(utc_time_col), 
                                          f.col('AIRPORT_TIMEZONE_DEST'))
                       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Aggregate 'hourly' flights
# MAGIC - Remove minutes portion 
# MAGIC   - For example, 8:15 -> 8:00 flight after transformation

# COMMAND ----------

def drop_minutes(flight_data):
    hourly_flights = flight_data.withColumn('CRS_DEP_TIME_UTC_HOUR'
                                            , f.date_trunc('hour', f.col('CRS_DEP_TIME_UTC')))\
                                .withColumn('CRS_ARR_TIME_UTC_HOUR'
                                            , f.date_trunc('hour', f.col('CRS_ARR_TIME_UTC')))
    return hourly_flights

flight_data = drop_minutes(flight_data)

# COMMAND ----------

# flight_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. Remove Unwanted Features

# COMMAND ----------

# Remove some columns we don't want
final_remv_columns =  [
     'DEP_TIME',
     'DEP_DELAY',
     'DEP_DELAY_NEW',
     'ARR_TIME',
     'ARR_DELAY',
     'ARR_DELAY_NEW',
     'ARR_DEL15',
     'ARR_DELAY_GROUP',
     'ARR_TIME_BLK',
     'ACTUAL_ELAPSED_TIME',
     'CARRIER_DELAY',
     'WEATHER_DELAY',
     'NAS_DELAY',
     'SECURITY_DELAY',
     'LATE_AIRCRAFT_DELAY',
     'DEP_TIME_UTC',
     'ARR_TIME_UTC'
]

# flight_data = flight_data.drop(*final_remv_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8. Drop US Territories Records
# MAGIC - Weather data only contains contiguous US states

# COMMAND ----------

us_territory_list = [
                     'PR', # Puerto Rico
                     'TT', # Trust Territories
                     'VI', # Virgin Islands
                     'GU', # Guam
                     'AS', # America Samoa
                     'MP'  # Northern Mariana Islands
                    ]

for territory in us_territory_list:
    flight_data = flight_data.where(
        (f.col('ORIGIN_STATE_ABR') != territory) & 
        (f.col('DEST_STATE_ABR') != territory)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Blob Azure Storage to Save Processed Data

# COMMAND ----------

# write data to blob
flight_data = main_data.filter(f.col('YEAR') == 2015)
flight_data.write.mode('overwrite').parquet(AIRLINE_2015_PROCESSED_PATH)

# COMMAND ----------

# read saved data from blob
flight_data = spark.read.parquet(AIRLINE_2015_PROCESSED_PATH).cache()

# COMMAND ----------

flight_data.count()

# COMMAND ----------

