# Databricks notebook source
# MAGIC %md
# MAGIC # Question Formulation
# MAGIC > _You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etc.. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics._
# MAGIC 
# MAGIC The goal is to predict delayed flights two hours ahead of departure time, thereby providing airlines and airports ample time to regroup and minimize the consequences as well as allowing passengers to be notified and increase customer satisfaction. A delay is defined as 15-minute delay (or greater) with respect to the planned time of departure.
# MAGIC 
# MAGIC ## Navigation
# MAGIC - [Data Load](#data-load)

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

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
# MAGIC ## Data Load
# MAGIC [Navigation](#navigation)

# COMMAND ----------

# Data Size
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
print(f'Total Data Size: {sum(total_size)/1e9:.2f} GB')

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/"))
display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/airlines_data/2015/"))
display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/"))

# COMMAND ----------



# COMMAND ----------

datasets_final_project_path = 'dbfs:/mnt/mids-w261/datasets_final_project'
parquet_airlines_data_path = datasets_final_project_path + '/parquet_airlines_data/'
parquet_airlines_data_3m_path = datasets_final_project_path + '/parquet_airlines_data_3m/'
parquet_airlines_data_6m_path = datasets_final_project_path + '/parquet_airlines_data_6m/'
stations_data_path = datasets_final_project_path + '/stations_data/'
weather_data_path = datasets_final_project_path + '/weather_data/'

df_airlines_data_3m = spark.read.parquet(parquet_airlines_data_3m_path + '*.parquet').cache() # 2015 Q1 flights
df_airlines_data_6m = spark.read.parquet(parquet_airlines_data_6m_path + '*.parquet').cache() # 2015 Q1+Q2 flights

main_data = df_airlines_data_3m

# COMMAND ----------

# df_airlines_data_6m.select(['DEST']).distinct().display()
df_airlines_data_6m.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA
# MAGIC ### Airline Data
# MAGIC #### 2015 6 months Dataset Description
# MAGIC - The mean delay is 14.24 minutes with standard deviation of 36.64 minutes with maximum delay time of 1221 minutes for `DEP_DELAY_NEW`.
# MAGIC - Since our main goal is to predict delays above 15 minutes, `DEP_DEL15` will be used.
# MAGIC - Only 21.91% of flights have delays above 15 minutes denoting an imbalanced dataset.
# MAGIC - Most features are categorical variables.

# COMMAND ----------

display(main_data)
display(main_data.describe())
delay_count = main_data.where('DEP_DEL15 > 0').count()
size = main_data.count()
print(f'Size of Airlines Data: {main_data.count():,}')
print(f'Percent of Significant Delays: {delay_count/size*100:.2f}%')

# COMMAND ----------

main_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Information
# MAGIC - A large number of records with invalid delay information such as departure or arrival time which correspond to diverted and cancelled flights.
# MAGIC - Inclusion of these data in will result in bias in our prediction.
# MAGIC - Fortunately, the distribution of these type of flights is pretty small and will be dropped during data cleaning.
# MAGIC - Diverted and cancelled flights have different impacts compared to delayed flights, as well as airports handling these differently.
# MAGIC - Creating a model to predict flight diversions and cancellations would likely require different features compared to predicting flight delays.

# COMMAND ----------

cancelled = main_data.where(f.col('CANCELLED') == 1).count()
diverted = main_data.where(f.col('DIVERTED') == 1).count()
print(f'Percent of Cancelled Flights: {cancelled/size*100:.4f}%')
print(f'Percent of Diverted Flights: {diverted/size*100:.4f}%')

# COMMAND ----------



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

df = main_data.toPandas()
null_df = null_values(df)

null_df

# COMMAND ----------

# Null percentages > 90%
null_df[null_df.Percentage > 90]

# COMMAND ----------

# Drop columns with more than 90% null values as these will be hard to impute
drop_columns_list = null_df[null_df.Percentage > 90].index.values.tolist()
drop_columns_list

# COMMAND ----------

# Reduced columns in Pandas df
df_reduced_cols = df.drop(drop_columns_list, axis=1)
df_reduced_cols.columns

# COMMAND ----------

# Reduced columns in Spark df
main_data_reduced_cols = main_data.drop(*drop_columns_list)
main_data_reduced_cols.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numeric Variables
# MAGIC - Compare correlation between numeric variables and `DEP_DELAY_NEW` in order to gain an idea on how these features are able to explain delay variances.
# MAGIC - Numeric variable criteria: `float64` or `double` data type and distinct values are more than 30

# COMMAND ----------

# Pandas
numeric_vars = [x for x in df_reduced_cols.columns if ('float64' in str(df_reduced_cols[x].dtypes)) and (len(df_reduced_cols[x].unique()) > 30)]

# Spark df
numeric_vars_spark = [x for x in main_data_reduced_cols.columns if ('double' in str(main_data_reduced_cols.select(x).schema).lower()) and (main_data_reduced_cols.select(x).distinct().count() > 30)]

numeric_vars_spark

# COMMAND ----------

# Heat map for numeric features
sns.set(rc={'figure.figsize':(20, 10)})
sns.heatmap(df_reduced_cols[numeric_vars_spark].corr(), cmap='RdBu_r', annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC - `ARR_*` and `DEP_*` variables have high correlation with `DEP_DELAY_NEW` 
# MAGIC   - However, since our task is to predict delays 2 hours prior to scheduled departure time, we will not be able to count on any information obtained after departure
# MAGIC   - For predictions, we can only use information 2 hours before scheduled departure time
# MAGIC - `LATE_AIRCRAFT_DELAY`(0.56), `CARRIER_DELAY`(0.51) and `WEATHER_DELAY`(0.42) have the highest correlation with `DEP_DELAY_NEW`
# MAGIC   - `LATE_AIRCRAFT_DELAY` - Delay caused by previous flight
# MAGIC   - `CARRIER_DELAY` - Delay caused by airlines (i.e. mechanical issues)
# MAGIC   - `WEATHER_DELAY` - Delay caused by bad weather
# MAGIC   - `NAS_DELAY` - Delay caused by airport operations due to weather or congestions
# MAGIC - The above information may be available more than 2 hours prior to scheduled departure

# COMMAND ----------

# Sort top predictors with 'DEP_DELAY_NEW'
sns.set(rc={'figure.figsize':(5, 10)})
sns.heatmap(df_reduced_cols[numeric_vars_spark].corr()[['DEP_DELAY_NEW']].abs().sort_values(by=['DEP_DELAY_NEW'], ascending=False), annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC  - Check for multicollinearity on delay factors
# MAGIC    - No great multicollinearity between delay factors

# COMMAND ----------

delay_factors = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
sns.set(rc={'figure.figsize':(10, 5)})
sns.heatmap(df_reduced_cols[delay_factors].corr(), cmap='RdBu_r', annot=True)

# COMMAND ----------

df_reduced_cols.loc[df_reduced_cols.DEP_DEL15 == 0, 'OUTCOME'] = 'On_Time'
df_reduced_cols.loc[df_reduced_cols.DEP_DEL15 == 1, 'OUTCOME'] = 'Delayed'
df_reduced_cols.loc[df_reduced_cols.CANCELLED == 1, 'OUTCOME'] = 'Cancelled'
ax = sns.countplot(x = df_reduced_cols.OUTCOME)
ax.bar_label(ax.containers[0]);
ax.set_title('Flight Outcome Counts');

# COMMAND ----------

main_data_reduced_cols = main_data_reduced_cols.withColumn('OUTCOME',
                                         f.when(f.col('DEP_DEL15') == 0, 'On_Time')
                                          .when(f.col('DEP_DEL15') == 1, 'Delayed')
                                          .when(f.col('CANCELLED') == 1, 'Cancelled'))
display(main_data_reduced_cols.groupBy('OUTCOME').count().sort('OUTCOME'))

# COMMAND ----------

# Drop cancelled flights
main_data_reduced_cols = main_data_reduced_cols.where(f.col('OUTCOME') != 'Cancelled')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Duplicate IDs
# MAGIC - Created a unique ID by combining the below:
# MAGIC   - `ID` - `FL_DATE`, `TAIL_NUM`, `ORIGIN`, `CRS_DEP_TIME`
# MAGIC   - Prevents 
# MAGIC - Upon unique ID creation, discovered duplicate flights
# MAGIC   - Some flights were cancelled and replaced by flights to a different destination
# MAGIC     - Remove cancelled flights and keep replacement flights
# MAGIC   - Some flights have same tail_num and origin but different destination
# MAGIC     - Remove the above as data seems suspicious

# COMMAND ----------

# MAGIC %md 
# MAGIC Before creating a unique id, we will make sure that the features we wish to concat on do not have any missing data

# COMMAND ----------

# Can see there are no null values here - so we can use this as a legitimate key to aggregate if needed
print(main_data_reduced_cols.where(f.col('FL_DATE').isNull()).count())
print(main_data_reduced_cols.where(f.col('TAIL_NUM').isNull()).count())
print(main_data_reduced_cols.where(f.col('ORIGIN').isNull()).count())
print(main_data_reduced_cols.where(f.col('CRS_DEP_TIME').isNull()).count())

# COMMAND ----------

unique_id = main_data_reduced_cols.withColumn('ID'
                                                        , f.concat(
                                                          f.col('FL_DATE')
                                                         ,f.col('TAIL_NUM')
                                                         ,f.col('ORIGIN')
                                                         ,f.col('CRS_DEP_TIME')
                                                        ))
display(unique_id)

# COMMAND ----------

# duplicate unique_ids

unique_id_count = unique_id.groupBy('ID').count()
duplicates = unique_id_count.filter(unique_id_count['count'] > 1)

display(duplicates)
display(unique_id.where(f.col('ID') == '2015-03-11N21197ORD845'))

# COMMAND ----------

# drop duplicate ids
dup_list = [x['ID'] for x in duplicates.select('ID').collect()]
unique_id2 = unique_id.filter(unique_id['ID'].isin(dup_list) == False)
unique_id2.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert to UTC Datetime
# MAGIC   - Airport Master: https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
# MAGIC   - Timezone: https://raw.githubusercontent.com/hroptatyr/dateutils/tzmaps/iata.tzmap

# COMMAND ----------

# Airport Coordinates
airport_coord = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)

# Rename headers
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
# Convert to spark df
airport_coord = spark.createDataFrame(airport_coord)

# Join on airport_master to airlines data based on origin
KEY_list = ['IATA', 'ICAO', 'AIRPORT_LAT', 'AIRPORT_LONG', 'AIRPORT_TIMEZONE', 'AIRPORT_UTC_OFFSET']
airport_coord_origin = unique_id2.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_ORIGIN'),
                                                            'ICAO', f.col('ICAO').alias('ICAO_ORIGIN'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_ORIGIN'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_ORIGIN'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_ORIGIN'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_ORIGIN')),
                                                            airport_coord.IATA == unique_id2.ORIGIN).drop(*KEY_list)
flight_data = airport_coord_origin.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_DEST'),
                                                             'ICAO', f.col('ICAO').alias('ICAO_DEST'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_DEST'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_DEST'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_DEST'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_DEST')),
                                                            airport_coord.IATA == unique_id2.DEST).drop(*KEY_list)


flight_data.display()

# COMMAND ----------

flight_data.columns

# COMMAND ----------

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
  
flight_data.select('CRS_DEP_TIME', 'CRS_DEP_TIME_UTC', 'AIRPORT_TIMEZONE_ORIGIN', 'AIRPORT_UTC_OFFSET_ORIGIN', 'CRS_ARR_TIME', 'CRS_ARR_TIME_UTC', 'AIRPORT_TIMEZONE_DEST', 'AIRPORT_UTC_OFFSET_DEST', 'MONTH').filter(flight_data.MONTH == 3).display()

# COMMAND ----------

# # flight_data.select(flight_data['AIRPORT_UTC_OFFSET_ORIGIN']).display()
# from pyspark.sql.types import * 
# def update_utc_offset(x):
#   if len(x)==2 and '-' in x:
#     x = '-'+'00'+'0'+x[1]
#   elif len(x)==3 and '-' in x:
#     x = '-'+'00'+x[1:]
#   elif len(x)==1:
#     x = '000'+x
#   elif len(x)==2:
#     x = '00'+x
#   else: 
#     print("I should not bhe here")
#   return x

# udf_star_desc = udf(update_utc_offset,StringType())
# flight_data_updated = flight_data.withColumn("UTC_OFFSET_UPDATE_ORIGIN",udf_star_desc(col("AIRPORT_UTC_OFFSET_ORIGIN")))
# flight_data_updated.select(concat_ws(' ',flight_data.FL_DATE,flight_data.DEP_TIME,flight_data_updated.UTC_OFFSET_UPDATE_ORIGIN).alias("concat")).withColumn('concat',to_timestamp("concat","yyyy-M-d HHmm X")).display()

# COMMAND ----------

# def get_utc_time(x):
#   x = x.split('  ')
#   iata_code = x[0]
#   time = x[1]
  
#   apt = airporttime.AirportTime(iata_code=iata_code)
#   dt = datetime.strptime(x[1], "%Y-%m-%d %H%M")
#   tz_aware_utc_time = apt.to_utc(dt)
#   return tz_aware_utc_time

# utc_offset_func = udf(get_utc_time,TimestampType())
# flight_data_updated = flight_data.withColumn("concat_iata_time", concat_ws('  ',flight_data.IATA_ORIGIN, flight_data.FL_DATE,flight_data.DEP_TIME)).withColumn("origin_time_updated",utc_offset_func(col("concat_iata_time")))
# flight_data_updated.display()
# #.alias("concat_iata_time")#.withColumn("origin_time_updated",utc_offset_func(col("concat_iata_time"))).display()


# COMMAND ----------

# Double-checking there are no duplicate IDs
# print(flight_data.count())
# print(flight_data.dropDuplicates(['ID']).count())

# COMMAND ----------

flight_data_h = flight_data.withColumn('CRS_DEP_TIME_UTC_HOUR', f.date_trunc('hour', f.col('CRS_DEP_TIME_UTC'))).withColumn('CRS_ARR_TIME_UTC_HOUR', f.date_trunc('hour', f.col('CRS_ARR_TIME_UTC')))
# display(flight_data_h.select(['CRS_DEP_TIME_UTC', 'CRS_DEP_TIME_UTC_HOUR']))

# COMMAND ----------

final_remv_columns =  ['DEP_TIME',
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
 'DIV_AIRPORT_LANDINGS',
 'OUTCOME',
 'DEP_TIME_UTC',
 'ARR_TIME_UTC']

# COMMAND ----------

flight_data_h = flight_data_h.drop(*final_remv_columns)

# COMMAND ----------

display(flight_data_h.select(['ICAO_ORIGIN', 'ICAO_DEST', 'CRS_ARR_TIME_UTC_HOUR', 'CRS_ARR_TIME_UTC_HOUR', 'ID']))
display(flight_data_h.select(f.col('ICAO_DEST')).distinct())

# COMMAND ----------

flight_data_pd_df = flight_data.toPandas()
null_values(flight_data_pd_df)

# COMMAND ----------

# # Airport Timezone
# airport_timezone = pd.read_csv("https://raw.githubusercontent.com/hroptatyr/dateutils/tzmaps/iata.tzmap", delimiter='\t', header=None)

# # Rename headers
# airport_timezone.rename(columns={0: 'IATA_DEST', 1: 'AIRPORT_TIMEZONE_DEST'},inplace=True)

# # Convert to spark df
# airport_timezone = spark.createDataFrame(airport_timezone).union(spark.createDataFrame([['XWA', 'America/Chicago']])) # add Williston, ND airport

# # Join on airport_timezone to flight data
# # flight_data_dest = 
# flight_data.join(airport_timezone, airport_timezone.IATA == flight_data.DEST).display()
# # flight_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

# Load the 2015 Q1 for Weather
# df_weather = spark.read.parquet(f'/mnt/mids-w261/datasets_final_project/weather_data/*').filter(col('DATE') < "2015-04-01T00:00:00.000")
# display(df_weather)

# df_weather_subset = df_weather.sample(False, 0.001, seed = 1)
# df_weather_subset.write.parquet(f'{blob_url}/weather_sample_30k')

df_weather_subset = spark.read.parquet(f'{blob_url}/weather_sample_30k').cache()
display(df_weather_subset)
size = df_weather_subset.count()
print(f'Size of Weather Data: {df_weather_subset.count():,}')

# COMMAND ----------

df_weather_subset.printSchema()

# COMMAND ----------

df = df_weather_subset.toPandas()
weather_null_df = null_values(df)

weather_null_df

# COMMAND ----------

weather_null_df[weather_null_df.Percentage > 50]

# COMMAND ----------

blank_dict = {}

# Find empty values
for field in df_weather_subset.columns:
  blank_dict[field] = df_weather_subset.select(field).where(f.col(field) == '').count()  

blank_df = pd.DataFrame([blank_dict])/size*100

# Plot
p_transpose = blank_df.T.values
plt.plot(p_transpose,'ro');
plt.title('Percentage of Blanks for each Feature',fontsize=18);
plt.xlabel('Feature Number',fontsize=15);
plt.ylabel('Percentage Blank', fontsize=15);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aggregate hourly weather data by station + Feature Selection
# MAGIC 
# MAGIC In preparation of our join to Airline data, we needed to aggregate weather data by time block. There is no guarantee that for each flight time we will have the exact weather timestamp that we need. Therefore, we will aggregate both to an hourly level - rounding down to the nearest hour. 
# MAGIC 
# MAGIC ##### Subsetting the data to include US weather stations
# MAGIC This sample (.1%) of the total weather data has 8368 Unique locations, many of them outside of the united states. Because the nature of our airline data is domestic - we can quickly reduce the size of the weather dataset by removing observations from outside of the US.
# MAGIC 
# MAGIC All of these names don't appear to follow the same format. For example, the above list in Japan appears to be City, country. Initial split `df_weather_subset.withColumn('Country', f.split(df_weather_subset['Name'], ',').getItem(1))` assuming this was true for all records was incorrect, as records are represented as 'City, ST Country'. Instead of splitting, we filter to all of the Names ending with "US". This reduced the amount of weather stations by about 25%. 
# MAGIC 
# MAGIC ##### Dealing with Null values in `STATION` and `NAME`
# MAGIC Before we can move on to EDA or aggregations, we have to properly treat `STATION` and `NAME`. If we have null values in these columns, it will disallow proper aggregation. 

# COMMAND ----------

FEATURES = eval(dbutils.notebook.run('/Users/rubyhan@berkeley.edu/team28/Final_Project/featureSelection', 60))

# Keep selected features
weather_full = df_weather_subset.select(FEATURES['WEATHER'])

# Keep US weather stations
df_weather_US = weather_full.filter(f.col('NAME').endswith('US'))

# Moving this to a lower cell after fixing the null issues with station and name
# df_weather_h = df_weather_US.withColumn('HOUR', f.date_trunc('hour', f.col('DATE')))\
#                             .withColumn('ROW_NUM', f.row_number().over(Window.partitionBy('NAME', 'HOUR')\
#                                                                  .orderBy(col('DATE').desc())))\
#                             .filter(f.col('ROW_NUM') == 1)\
#                             .drop('ROW_NUM')

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


display(df_weather_US_temp)

# COMMAND ----------

# FEATURES = eval(dbutils.notebook.run('/Users/rubyhan@berkeley.edu/team28/Final_Project/featureSelection', 60))
# df_weather_2015_Q1 = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-07-01T00:00:00.000")
# # Keep selected features
# weather_full = df_weather_2015_Q1.select(FEATURES['WEATHER'])

# # Keep US weather stations
# df_weather_US = weather_full.filter(f.col('NAME').endswith('US'))

# COMMAND ----------

# # First, create a mapping table that take all non-null station ids and maps them to all non-null station names:
# name_to_station = df_weather_US.where((col('NAME').isNull() == False) & (col('STATION').isNull() == False))\
#                                .select('NAME', 'STATION')\
#                                .withColumnRenamed('STATION', 'REP_STATION')\
#                                .withColumnRenamed('NAME', 'REP_NAME')\
#                                .distinct()

# # First, need to focus on replacing null weather stations with known station id. we do this by creating a new column of non-null stations via join, then replacing station with rep_station (which is non-null)
# df_weather_US_temp = df_weather_US.join(name_to_station, df_weather_US.NAME == name_to_station.REP_NAME, "left")\
#                                   .select(df_weather_US['*'], name_to_station['REP_STATION'])\
#                                   .withColumn('STATION', f.when(f.col('STATION').isNull(), 'REP_STATION')\
#                                                               .otherwise(f.col('STATION')))\
#                                                               .drop('REP_STATION')

# # Second, need to focus on replacing null weather stations with known name. we do this by creating a new column of non-null names via join, then replacing station with rep_name (which is non-null)
# df_weather_US_temp = df_weather_US_temp.join(name_to_station, df_weather_US_temp.STATION == name_to_station.REP_STATION, "left")\
#                                   .select(df_weather_US_temp['*'], name_to_station['REP_NAME'])\
#                                   .withColumn('NAME', f.when(f.col('NAME').isNull(), 'REP_NAME')\
#                                                               .otherwise(f.col('NAME')))\
#                                                               .drop('REP_NAME')
# display(df_weather_US_temp)


# COMMAND ----------

# Check to see if we've fixed the null values issue with `NAME`
display(df_weather_US_temp.where(col('NAME').isNull()))

# COMMAND ----------

# Check to see if we've fixed the null values issue with `station`
display(df_weather_US_temp.where(col('STATION').isNull()))

# COMMAND ----------

# Writing this to parquet so we can use it later without having to re-run
# df_weather_US_2015_Q1_filled_null_stations = df_weather_US_temp
df_weather_US_2015_Q1_filled_null_stations.write.parquet(f'{blob_url}/weather_filled_2015_Q1')
# df_weather_US_2015_Q1 = spark.read.parquet(f'{blob_url}/weather_filled_2015_Q1').cache()

# COMMAND ----------

display(df_weather_US_temp.where((col('STATION').isNull()) | (col('NAME').isNull())))


# COMMAND ----------

# MAGIC %md
# MAGIC Now that all of the null stations/names have been resolved - we can key off of them to do some aggregation. 

# COMMAND ----------

df_weather_h = df_weather_US_temp.withColumn('HOUR', f.date_trunc('hour', f.col('DATE')))\
                            .withColumn('ROW_NUM', f.row_number().over(Window.partitionBy('STATION', 'HOUR')\
                                                                 .orderBy(col('DATE').desc())))\
                            .filter(f.col('ROW_NUM') == 1)\
                            .drop('ROW_NUM')

# COMMAND ----------

# MAGIC %md
# MAGIC Now inspecing a particular airport that had some duplicate/missing values - we can see that the logic is funcitioning properly - ROW_NUM is assigned properly across STATION, and we will next filter on ROW_NUM = 1, which pulls the last observation over a given time block

# COMMAND ----------

display(df_weather_h.where(col('NAME') == 'CARL R KELLER FIELD AIRPORT, OH US').orderBy(col('HOUR')))

# COMMAND ----------

# Run the final time aggregation - only keeping most recent observation
df_weather_h = df_weather_US_temp.withColumn('HOUR', f.date_trunc('hour', f.col('DATE')))\
                            .withColumn('ROW_NUM', f.row_number().over(Window.partitionBy('STATION', 'HOUR')\
                                                                 .orderBy(col('DATE').desc())))\
                            .filter(f.col('ROW_NUM') == 1)\
                            .drop('ROW_NUM')

# COMMAND ----------

# MAGIC %md 
# MAGIC Following hourly weather aggregation, we inspect our tables, and can see that things are looking as expected. There are still a very small portion of observations that do not have station ID, and these are ones that did not appear in our sample with a valid name_to_station mapping. in the larger datasets we expect this will be resolved.

# COMMAND ----------

display(df_weather_h.orderBy('DATE'))
print(df_weather_h.count())
df_weather_h.filter(col('STATION').isNull()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Six of our features in weather are concatenations of a few different metrics - we will break these down into their relevant measures for both search/treatment of null values, as well as allowing the model to properly use this data. 

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

# COMMAND ----------

df_weather_subset_decoded = weather_decode(df_weather_h)
df_weather_subset_decoded.display()

# COMMAND ----------

# # Drop columns with more than 50% null values as these will be hard to impute
# drop_columns_list = weather_null_df[weather_null_df.Percentage > 50].index.values.tolist()
# drop_columns_list

# COMMAND ----------

# # Reduced columns in Spark df
# unmeaningful_cols = ['SOURCE', 'CALL_SIGN', 'QUALITY_CONTROL', 'GA1', 'GF1', 'MA1', 'REM']

# df_weather_subset_decoded_reduced_cols = df_weather_subset_decoded.drop(*drop_columns_list
#                                                                        + unmeaningful_cols)
# df_weather_subset_decoded_reduced_cols.columns
df_weather_subset_decoded_reduced_cols = df_weather_subset_decoded

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have fixed our null value issue for our primary key - we can move on to EDA. Let's look at some figures for data before and after we treat null values in other columns. 
# MAGIC 
# MAGIC #### Descriptive EDA
# MAGIC We've started here by removing many of the weather features that we've discovered have extremely high representation of nulls. Further exploration on these columns is shown [here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885900748/command/1038316885900773). 

# COMMAND ----------

df_eda_pre_treat = df_weather_subset_decoded_reduced_cols.toPandas()
df_eda_pre_treat.describe() # describe the numeric variables

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Descriptive EDA for remaining Features
# MAGIC Following removal of features with disproportionate representation of nulls, we dig further into our remaining features to understand data quality. We know from [previous eda on a larger version of this data](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885900748/command/1038316885900784) that the below columns do not contain any 'Null', blanks, or NaNs, but are there other 'codes' used to represent missing data? We can quickly see from our set of histograms (and references in the codebook) that this dataset commonly leverages some versoin of repeating 9's as a missing data indicator. 
# MAGIC 
# MAGIC Features without any 'traditional' Null values: 
# MAGIC - Date
# MAGIC - Latitude
# MAGIC - Longitude
# MAGIC - Elevation
# MAGIC - WND
# MAGIC - CIG
# MAGIC - VIS
# MAGIC - TMP
# MAGIC - DEW
# MAGIC - SLP

# COMMAND ----------

import math

def histograms(df_in, variables = []):
    if len(variables) == 0:
        variables = df_in.columns
    chart_size = math.ceil(len(variables)**.5)
        
    fig = plt.figure(figsize = (20, 20))

    for i, var in enumerate(variables):
        ax = fig.add_subplot(chart_size, chart_size, i+1)
        ax.set_title(var + '\n max value:' + str(max(df_in[var])))
        df_in[var].hist(ax = ax)
    fig.tight_layout()
    plt.show()

# COMMAND ----------

weather_US.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC First, we need to separate out numerical and non-numerical features, as our research will flow a bit differently here. 

# COMMAND ----------


# Separating numeric and non-numeric Columns
feat_types = df_eda_pre_treat.dtypes
non_num_feats = feat_types[(feat_types != 'float64') & (feat_types != 'int32')].index
num_feats = feat_types.index.drop(non_num_feats)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use the below histograms as a first look at some of the different codes that are used to represent missing values. For example, `WND_DIRECTION_ANGLE` uses '999', whereas `WND_SPEED_RATE` uses '9999'. 
# MAGIC 
# MAGIC Here, we are specifically interested in both identifying the 'missing' code, as well as understanding what portion of each feature contains missing data. this will further influence whether we drop observations, treat observations, or drop further features. Description of each of our numerical variables is included below. 
# MAGIC 
# MAGIC `LATITUDE` (double): Latitude coordinates
# MAGIC - Missing = 999999 (none missing in sample)
# MAGIC - Range = (-90000, 90000)
# MAGIC - Scaling Factor = 1000
# MAGIC 
# MAGIC `LONGITUDE` (double): Longitude coordinates
# MAGIC - Missing = 999999 (none missing in sample)
# MAGIC - Range = (-90000, 90000)
# MAGIC - Scaling Factor = 1000
# MAGIC 
# MAGIC `ELEVATION` (double): The elevation of a geophysical-point-observation relative to Mean Sea Level (MSL).
# MAGIC - Missing = 9999 (none missing in sample)
# MAGIC - Range = (-0400, 8850)
# MAGIC - Scaling Factor = 1
# MAGIC 
# MAGIC **WND** Derivatives - Transformed
# MAGIC 
# MAGIC `WND_DIRECTION_ANGLE`: The angle, measured in a clockwise direction, between true north and the direction from which the wind is blowing.
# MAGIC - Missing = 999 (% Missing in sample)
# MAGIC - Range = (001, 360)
# MAGIC - Scaling Factor = 1
# MAGIC 
# MAGIC `WND_SPEED_RATE`: The rate of horizontal travel of air past a fixed point.
# MAGIC - Missing = 9999
# MAGIC - Range = (0000, 0900), categorical - See code book pg 8 for category definitions
# MAGIC - Units: meters per second
# MAGIC - Scaling Factor = 10
# MAGIC 
# MAGIC **CIG** Derivatives - Transformed
# MAGIC 
# MAGIC `CIG_CEILING_HEIGHT_DIMENSION`: The height above ground level (AGL) of the lowest cloud or obscuring phenomena layer aloft with 5/8 or more summation total sky cover, which may be predominantly opaque, or the vertical visibility into a surface-based obstruction. 
# MAGIC - Missing = 99999 (% Missing in sample)
# MAGIC - Range = (00000, 22000)
# MAGIC - Units = meters
# MAGIC - Scaling Factor = 1
# MAGIC 
# MAGIC **VIS** Derivatives - Transformed
# MAGIC 
# MAGIC `VIS_DISTANCE DIMENSION`: The horizontal distance at which an object can be seen and identified
# MAGIC - Missing = 999999 (% Missing in sample)
# MAGIC - Range = (000000, 160000) - values greater than 160000 entered as 160000
# MAGIC - Units = meters
# MAGIC - Scaling Factor = 1
# MAGIC 
# MAGIC `VIS_QUALITY_VARIABILITY_CODE`: The code that denotes whether or not the reported visibility is variable.
# MAGIC - Missing = 9 (% Missing in sample)
# MAGIC - Range = (000000, 160000) - values greater than 160000 entered as 160000
# MAGIC - Units = meters
# MAGIC - Scaling Factor = 1
# MAGIC 
# MAGIC **TMP** Derivatives - Transformed
# MAGIC 
# MAGIC `TMP_AIR_TEMP`: The temperature of the air.
# MAGIC - Missing = 9999 (% Missing in sample)
# MAGIC - Range = (-0932, 0618) - values greater than 160000 entered as 160000
# MAGIC - Units = Degrees Celsius
# MAGIC - Scaling Factor = 10
# MAGIC 
# MAGIC **DEW** Derivatives - Transformed
# MAGIC 
# MAGIC `DEW_POINT_TEMP`: The temperature to which a given parcel of air must be cooled at constant pressure and water vapor content in order for saturation to occur.
# MAGIC - Missing = 9999 (% Missing in sample)
# MAGIC - Range = (-0932, 0368) - values greater than 160000 entered as 160000
# MAGIC - Units = Degrees Celsius
# MAGIC - Scaling Factor = 10
# MAGIC 
# MAGIC **SLP** Derivatives - Transformed
# MAGIC 
# MAGIC `SLP_SEA_LEVEL_PRES`: The air pressure relative to Mean Sea Level (MSL).
# MAGIC - Missing = 99999 (% Missing in sample)
# MAGIC - Range = (08600, 10900) - values greater than 160000 entered as 160000
# MAGIC - Units = Hectopascals
# MAGIC - Scaling Factor = 10
# MAGIC 
# MAGIC `_QUALITY_CODE`s: denotes the quality status of a given measure
# MAGIC - 0 = Passed gross limits check
# MAGIC - 1 = Passed all quality control checks
# MAGIC - **2 = Suspect**
# MAGIC - **3 = Erroneous**
# MAGIC - 4 = Passed gross limits check, data originate from an NCEI data source
# MAGIC - 5 = Passed all quality control checks, data originate from an NCEI data source 6 = Suspect, data originate from an NCEI data source
# MAGIC - **7 = Erroneous, data originate from an NCEI data source**
# MAGIC - 9 = Passed gross limits check if element is present
# MAGIC 
# MAGIC - Note: 2, 3, and 7 should be used in conjunction with missing codes to either impute or remove datapoints

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have an idea of these different Null representations, let's look at what % of each of our values are actually missing, not just traditional 'nulls'

# COMMAND ----------

features = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'WND_DIRECTION_ANGLE', 'WND_SPEED_RATE', 'CIG_CEILING_HEIGHT_DIMENSION', 'VIS_DISTANCE_DIMENSION', 'VIS_QUALITY_VARIABILITY_CODE', 'TMP_AIR_TEMP', 'DEW_POINT_TEMP', 'SLP_SEA_LEVEL_PRES']
missing_codes = [999999, 999999, 9999, 999, 9999, 99999, 999999, 9, 9999, 9999, 99999]

missing_pct = []
for i, feat in enumerate(features): 
    pct = len(df_eda_pre_treat[df_eda_pre_treat[feat] == missing_codes[i]])/len(df_eda_pre_treat)
    missing_pct.append(pct)
    
missing_pct



pd.DataFrame(missing_pct, index = features)

# COMMAND ----------

histograms(df_eda_pre_treat, num_feats)

# COMMAND ----------

# MAGIC %md
# MAGIC Next looking at non-numeric features, we can see there is also a high proportion of missing data, represented by a code of '9'

# COMMAND ----------

categorical_feats = non_num_feats[3:]
for feat in categorical_feats:
    print('Categorical Feature: ' + feat)
    print(df_eda_pre_treat[feat].value_counts()/df_eda_pre_treat[feat].count())
    print('---------------------------------')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC The high proportion of misging weather data motivates treatment - we still want to be able to use some of this data - and would prefer to not drop an entire row because one column is null. There are two ways we tackle this: 
# MAGIC 1. Removing invalid data
# MAGIC     a. we know from the `_QUALITY` features that there are two types of measurements that represent unusable data - codes of '3' and codes of '7'
# MAGIC     b. Additionally, some data fall outside of the relevant ranges defined by the codebook
# MAGIC     c. Filtering out these two data types will reduce the amount of erroneous data we include, and will also hopefully reduce our null values - as we believe there is correlation between the two. 
# MAGIC 2. Imputing null data
# MAGIC 
# MAGIC #### Remove Invalid Weather Data
# MAGIC - Invalid data will be removed and is defined as not within range based on documentation including missing information.
# MAGIC - All error values will be kept as they may indicate instrument error due to extreme weather conditions.

# COMMAND ----------

df_weather_subset_decoded_reduced_cols

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
  df_weather = df_weather_subset_decoded_reduced_cols.withColumn(col, f.col(col).cast('int'))

df_weather = df_weather_subset_decoded_reduced_cols.filter(
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

df_weather.count()

# COMMAND ----------

df_weather.filter(df_weather.WND_DIRECTION_ANGLE == 999).display()

# COMMAND ----------

df_weather.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute Null Weather Data
# MAGIC - Replace missing info with null and impute hourly missing data based on 7-day average

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
                                                                             
# df_weather_valid_fillna.select('STATION', 'DATE', 'HOUR', 'WND_DIRECTION_ANGLE', 'WND_DIRECTION_ANGLE-AVG').orderBy('STATION','DATE').display()

null_comparison = df_weather_valid_fillna.select([f.count(f.when(f.col(c).contains('None') | \
                            f.col(c).contains('NULL') | \
                            (f.col(c) == '' ) | \
                            f.col(c).isNull(), c 
                           )).alias(c)
                    for c in df_weather_valid_fillna.columns])

null_comparison.display()

# COMMAND ----------

# MAGIC %md
# MAGIC After imputing the data with a rolling 7-day average, we unfortunately still see quite a few missing datapoints. After further research, we are attributing this to the fact that we are using a .1% sample of the data. This measn that a lot of continuous data is missing - so when our imputation function looks back to find data within 7 days - in many cases it cannot find anything. We have excluded a lot of our data for the sake of unit testing. This will need to be figured out after running on the larger datasets. 

# COMMAND ----------

display(null_comparison)

# COMMAND ----------

null_eda = null_comparison.toPandas()
df_eda_post_treat = df_weather_valid_fillna.toPandas()
null_eda.T.sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've imputed our missing data - lets take a look at some of the key EDA we ran before the imputations. This will now be run on `df_eda_post_treat`

# COMMAND ----------

df_eda_post_treat.describe()

# COMMAND ----------

df_eda_post_treat.columns

# COMMAND ----------

imputed_feats = ['LATITUDE', 'LONGITUDE', 'ELEVATION',
       'WND_DIRECTION_ANGLE', 'WND_SPEED_RATE',
       'CIG_CEILING_HEIGHT_DIMENSION', 'VIS_DISTANCE_DIMENSION', 
       'TMP_AIR_TEMP', 'DEW_POINT_TEMP', 'SLP_SEA_LEVEL_PRES']

# COMMAND ----------

histograms(df_eda_post_treat, imputed_feats)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_weather_valid_fillna.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations Data

# COMMAND ----------

df_stations = spark.read.parquet('/mnt/mids-w261/datasets_final_project/stations_data/*').cache()
display(df_stations)

# COMMAND ----------

# df_stations.count()

# COMMAND ----------

df_stations.select(f.col('neighbor_call')).distinct().display()

# COMMAND ----------

df_stations.where(f.col('neighbor_call') == 'KATL').orderBy(f.col('distance_to_neighbor')).display()

# COMMAND ----------

df_stations.where(f.col('station_id') == '70104526649').display()

# COMMAND ----------

df_stations.where(f.col('distance_to_neighbor').isNull()).display()

# COMMAND ----------

# station_ICOA_distance = df_stations.select(['neighbor_call', 'station_id', 'distance_to_neighbor']).where(f.col('neighbor_call')== "KOAR").count()
station_ICOA_distance = df_stations.select(['neighbor_call', 'station_id', 'distance_to_neighbor']).orderBy(['neighbor_call','distance_to_neighbor'])
station_ICOA_distance.display()

# COMMAND ----------

station_ICOA_distance_temp = station_ICOA_distance.groupby(f.col('neighbor_call')).min('distance_to_neighbor')
display(station_ICOA_distance_temp)

# COMMAND ----------

station_to_ICOA = station_ICOA_distance.withColumn('row_number', f.row_number()\
                                 .over(Window.partitionBy(f.col('neighbor_call'))\
                                 .orderBy(f.col('distance_to_neighbor').asc())))\
                                 .where(f.col('row_number') == 1)\
#                                  .drop(f.col('row_number'))
station_to_ICOA.display()

# COMMAND ----------

# val windowedPartition = Window.partitionBy(f.col('station_id')).orderBy(f.col('distance_to_neighbor').asc)
station_to_ICOA = station_ICOA_distance.withColumn('row_number', f.row_number()\
                                 .over(Window.partitionBy(f.col('neighbor_call'))\
                                 .orderBy(f.col('distance_to_neighbor').asc())))\
                                 .where(f.col('row_number') == 1)\
                                 .drop(f.col('row_number'))
station_to_ICOA.display()

# COMMAND ----------

# Ensure there is no data loss
station_to_ICOA.select(['neighbor_call']).distinct().count() == df_stations.select(['neighbor_call']).distinct().count()

# COMMAND ----------

weather_stations = df_weather_valid_fillna.select(['STATION']).distinct()

# COMMAND ----------

flight_ICAOs = flight_data_h.select(['ICAO_DEST']).distinct()

# COMMAND ----------

# df_weather_valid_fillna
# flight_data_h

weather_stations.join(station_to_ICOA, weather_stations.STATION == station_to_ICOA.station_id,'left').where(f.col('station_id').isNull()).display()

# COMMAND ----------

df_weather_valid.where(f.col('STATION')=='99999925711').display()

# COMMAND ----------

flight_ICAOs.join(station_to_ICOA, flight_ICAOs.ICAO_DEST == station_to_ICOA.neighbor_call,'left').where(f.col('neighbor_call').isNull()).display()

# COMMAND ----------

flight_data_h.select(f.col('ICAO_DEST')).distinct().display()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA Following Joins
# MAGIC 
# MAGIC Before moving on to a large Join, we want to make use of our 3 month join as it allows some easy work in pandas - before tha data is too large to fit into this framework

# COMMAND ----------

#Load data for two different join sample sizes
# %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports
# JOINED_2015_PROCESSED_PATH = blob_url + '/processed/joined_2015_data.parquet'
# JOINED_3M_PROCESSED_PATH = blob_url + '/processed/joined_3m_data.parquet'

flights_3m = spark.read.parquet(JOINED_3M_PROCESSED_PATH).cache()
flights_2015 = spark.read.parquet(JOINED_2015_PROCESSED_PATH).cache()

# COMMAND ----------

FEATURE_SELECTED = set(['YEAR','QUARTER','MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'OP_UNIQUE_CARRIER','ORIGIN_AIRPORT_ID','ORIGIN','DEST_AIRPORT_ID','DEST','DEP_DEL15','DEP_TIME_BLK',
'ARR_TIME_BLK','CRS_ELAPSED_TIME','ID','AIRPORT_LAT_ORIGIN','AIRPORT_LONG_ORIGIN','AIRPORT_LAT_DEST','AIRPORT_LONG_DEST','CRS_DEP_TIME_UTC_HOUR','CRS_DEP_TIME','ORIGIN_WEATHER_ELEVATION',
                        'ORIGIN_WEATHER_WND_DIRECTION_ANGLE-AVG','ORIGIN_WEATHER_WND_SPEED_RATE-AVG','ORIGIN_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG','ORIGIN_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
                      'ORIGIN_WEATHER_TMP_AIR_TEMP-AVG','ORIGIN_WEATHER_DEW_POINT_TEMP-AVG','ORIGIN_WEATHER_SLP_SEA_LEVEL_PRES-AVG','DEST_WEATHER_WND_DIRECTION_ANGLE-AVG','DEST_WEATHER_WND_SPEED_RATE-AVG','DEST_WEATHER_ELEVATION','DEST_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG','DEST_WEATHER_VIS_DISTANCE_DIMENSION-AVG','DEST_WEATHER_TMP_AIR_TEMP-AVG','DEST_WEATHER_DEW_POINT_TEMP-AVG','DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG'])

flights_3m = flights_3m.select(*FEATURE_SELECTED)

len(flights_3m.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC Examining Null values on our 3M sample, we can see that we will have to drop at least 2.3% of our rows as we still have a few null values remaining. 

# COMMAND ----------

null_df = null_values(flights_3m.toPandas())
null_df

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's check how many row's we'd be removing on our 2015 dataset. Due to the size, we are unable to load it into pandas . INstead, we will remove null rows, then compare DF sizes. 

# COMMAND ----------

flights_2015 = flights_2015.select(*FEATURE_SELECTED)

# COMMAND ----------

full_count = flights_2015.count()
non_na_rows = flights_2015.na.drop().count()
na_rows = full_count - non_na_rows
print(f'{na_rows} rows contained a null value ({na_rows/full_count*100:.2f}%)')

# COMMAND ----------

# MAGIC %md
# MAGIC Running a few final checks to make sure our nulls appear not to have some pattern to them

# COMMAND ----------



# COMMAND ----------

null_by_airline = flights_2015.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()) \
                              .groupby('OP_UNIQUE_CARRIER')\
                              .count()\
                              .toPandas()

null_by_day = flights_2015.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()) \
                              .groupby('DAY_OF_WEEK')\
                              .count()\
                              .toPandas()

null_by_time_blk = flights_2015.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()) \
                              .groupby('DEP_TIME_BLK')\
                              .count()\
                              .toPandas()

null_by_month = flights_2015.where(f.col('DEST_WEATHER_WND_SPEED_RATE-AVG').isNull()) \
                              .groupby('MONTH')\
                              .count()\
                              .toPandas()

# COMMAND ----------

# Plot count of nulls by month
null_by_airline.set_index('OP_UNIQUE_CARRIER').sort_values('OP_UNIQUE_CARRIER').plot.bar()

# Plot count of nulls by month
null_by_day.set_index('DAY_OF_WEEK').sort_values('DAY_OF_WEEK').plot.bar()

# Plot count of nulls by month
null_by_time_blk.set_index('DEP_TIME_BLK').sort_values('DEP_TIME_BLK').plot.bar()

# Plot count of nulls by month
null_by_month.set_index('MONTH').sort_values('MONTH').plot.bar()

# COMMAND ----------

