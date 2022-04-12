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

from pyspark.sql import functions as f
from pyspark.sql.functions import col, to_timestamp, to_utc_timestamp, concat_ws

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
KEY_list = ['IATA', 'AIRPORT_LAT', 'AIRPORT_LONG', 'AIRPORT_TIMEZONE', 'AIRPORT_UTC_OFFSET']
airport_coord_origin = unique_id2.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_ORIGIN'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_ORIGIN'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_ORIGIN'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_ORIGIN'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_ORIGIN')),
                                                            airport_coord.IATA == unique_id2.ORIGIN).drop(*KEY_list)
flight_data = airport_coord_origin.join(airport_coord.select('IATA', f.col('IATA').alias('IATA_DEST'),
                                                            'AIRPORT_LAT', f.col('AIRPORT_LAT').alias('AIRPORT_LAT_DEST'),
                                                            'AIRPORT_LONG', f.col('AIRPORT_LONG').alias('AIRPORT_LONG_DEST'),
                                                            'AIRPORT_TIMEZONE', f.col('AIRPORT_TIMEZONE').alias('AIRPORT_TIMEZONE_DEST'),
                                                            'AIRPORT_UTC_OFFSET', f.col('AIRPORT_UTC_OFFSET').alias('AIRPORT_UTC_OFFSET_DEST')),
                                                            airport_coord.IATA == unique_id2.DEST).drop(*KEY_list)


flight_data.display()

# COMMAND ----------

time_list = [flight_data.CRS_DEP_TIME, flight_data.ARR_TIME, flight_data.DEP_TIME, flight_data.CRS_ARR_TIME]
utc_time_list = ['CRS_DEP_TIME_UTC', 'ARR_TIME_UTC', 'DEP_TIME_UTC', 'CRS_ARR_TIME_UTC']

for time_col, utc_time_col in zip(time_list, utc_time_list):
  flight_data = flight_data.withColumn(utc_time_col, 
                       f.to_timestamp(
                         f.concat(
                           f.col('FL_DATE'),
                           f.lpad(time_col, 4, '0')
                         ), format='yyyy-MM-ddHHmm'
                       ))\
            .withColumn(utc_time_col,
                       f.to_utc_timestamp(f.col(utc_time_col), 
                                          f.when(f.lit('DEP').isin(time_col),
                                          f.col('AIRPORT_TIMEZONE_ORIGIN'))\
                                          .otherwise(f.col('AIRPORT_TIMEZONE_DEST')))
                       )
  
flight_data.select('CRS_DEP_TIME', 'CRS_DEP_TIME_UTC', 'AIRPORT_TIMEZONE_ORIGIN', 'AIRPORT_UTC_OFFSET_ORIGIN', 'CRS_ARR_TIME', 'CRS_ARR_TIME_UTC', 'AIRPORT_TIMEZONE_DEST', 'AIRPORT_UTC_OFFSET_DEST').display()

# COMMAND ----------

flight_data_pd_df = flight_data.toPandas()
null_values(flight_data_pd_df)

# COMMAND ----------

# flight_data.select([])
from pyspark.sql.functions import concat,concat_ws,to_timestamp, to_utc_timestamp

# flight_data.select(concat_ws(' ',flight_data.FL_DATE,flight_data.DEP_TIME, flight_data.AIRPORT_TIMEZONE_ORIGIN).alias("concat"))#.display().withColumn("concat_2", to_utc_timestamp(col("concat"), col("AIRPORT_TIMEZONE_ORIGIN"))).display()

flight_data.select(concat_ws(' ',flight_data.FL_DATE,flight_data.DEP_TIME,flight_data.AIRPORT_UTC_OFFSET_ORIGIN).alias("concat")).withColumn('concat',to_timestamp("concat","yyyy-MM-dd HHmm z")).display()#.withColumn("concat_2", to_utc_timestamp(col("concat"),col("AIRPORT_TIMEZONE_ORIGIN"))).display()

#.withColumn('concat',to_timestamp("concat","yyyy-MM-dd HHmm %z")).display()
# flight_data.select(concat_ws(' ',flight_data.FL_DATE,flight_data.DEP_TIME).alias("concat")).withColumn('concat',to_timestamp("concat","yyyy-MM-dd HHmm")-flight_data.AIRPORT_UTC_OFFSET_ORIGIN).display()

# COMMAND ----------

dt = datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")

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

# MAGIC %md
# MAGIC Weather data has 8368 Unique locations, many of them outside of the united states. Because the nature of our airline data is domestic - we can quickly reduce the size of the weather dataset by removing observations from outside of the US.

# COMMAND ----------

df_weather_subset.select('Name').distinct().show(10)
cnt_distinct = df_weather_subset.select(f.col('Name')).distinct().count()
print(f'Number of distinct weather stations: {cnt_distinct}')

# COMMAND ----------

# MAGIC %md
# MAGIC All of these names don't appear to follow the same format. For example, the above list in Japan appears to be City, country. Initial split `df_weather_subset.withColumn('Country', f.split(df_weather_subset['Name'], ',').getItem(1))` assuming this was true for all records was incorrect, as records are represented as 'City, ST Country'. Instead of splitting, we filter to all of the `Name`s ending with "US". This reduced the amount of weather stations from 8300+ to 2278, and total rows from the quarter by %. 

# COMMAND ----------

df_weather_subset_US = df_weather_subset.filter(f.col('Name').endswith('US'))
print('count of US weatherstations', df_weather_subset_US.select(f.col('Name')).distinct().count())
print(df_weather_subset.count())
print(df_weather_subset_US.count())

# COMMAND ----------

df_weather_subset.printSchema()

# COMMAND ----------

df = df_weather_subset.toPandas()
weather_null_df = null_values(df)

weather_null_df

# COMMAND ----------



# COMMAND ----------

weather_null_df[weather_null_df.Percentage > 60]

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
       )
  
  return df

# COMMAND ----------

df_weather_subset_decoded = weather_decode(df_weather_subset)
display(df_weather_subset_decoded)

# COMMAND ----------

# Drop columns with more than 50% null values as these will be hard to impute
drop_columns_list = weather_null_df[weather_null_df.Percentage > 50].index.values.tolist()
drop_columns_list

# Consider dropping GA1 bc we have CIG, drop 

# COMMAND ----------

# Reduced columns in Spark df
unmeaningful_cols = ['SOURCE', 'CALL_SIGN', 'QUALITY_CONTROL', 'GA1', 'GF1', 'MA1', 'REM']
df_weather_subset_decoded_reduced_cols = df_weather_subset_decoded.drop(*drop_columns_list
                                                                        + ['WND',
                                                                           'CIG',
                                                                           'VIS',
                                                                           'TMP',
                                                                           'DEW',
                                                                           'SLP'
                                                                          ] + unmeaningful_colsunmeaningful_cols
                                                                       )
df_weather_subset_decoded_reduced_cols.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subsetting the data further to only include US weather stations
# MAGIC 
# MAGIC This sample (.1%) of the total weather data has 8368 Unique locations, many of them outside of the united states. Because the nature of our airline data is domestic - we can quickly reduce the size of the weather dataset by removing observations from outside of the US.
# MAGIC 
# MAGIC All of these names don't appear to follow the same format. For example, the above list in Japan appears to be City, country. Initial split `df_weather_subset.withColumn('Country', f.split(df_weather_subset['Name'], ',').getItem(1))` assuming this was true for all records was incorrect, as records are represented as 'City, ST Country'. Instead of splitting, we filter to all of the Names ending with "US". This reduced the amount of weather stations by about 25%. 
# MAGIC 
# MAGIC Next, we will quickly convert our spark dataframe to pandas for some graphical EDA. 

# COMMAND ----------

weather_US = df_weather_subset_decoded_reduced_cols.filter(f.col('Name').endswith('US'))
weather_eda = weather_US.toPandas()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Descriptive EDA for remaining Features
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
feat_types = weather_eda.dtypes
non_num_feats = feat_types[(feat_types != 'float64') & (feat_types != 'int32')].index
num_feats = feat_types.index.drop(non_num_feats)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use these histograms as a first look at some of the different codes that are used to represent missing values. For example, `WND_DIRECTION_ANGLE` uses '999', whereas `WND_SPEED_RATE` uses '9999'. 
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

features = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'WND_DIRECTION_ANGLE', 'WND_SPEED_RATE', 'CIG_CEILING_HEIGHT_DIMENSION', 'VIS_DISTANCE_DIMENSION', 'VIS_QUALITY_VARIABILITY_CODE', 'TMP_AIR_TEMP', 'DEW_POINT_TEMP', 'SLP_SEA_LEVEL_PRES']
missing_codes = [999999, 999999, 9999, 999, 9999, 99999, 999999, 9, 9999, 9999, 99999]

missing_pct = []
for i, feat in enumerate(features): 
    pct = len(weather_eda[weather_eda[feat] == missing_codes[i]])/len(weather_eda)
    missing_pct.append(pct)
    
missing_pct



pd.DataFrame(missing_pct, index = features)

# COMMAND ----------

histograms(weather_eda, num_feats)

# COMMAND ----------

categorical_feats = non_num_feats[3:]
for feat in categorical_feats:
    print('Categorical Feature: ' + feat)
    print(weather_eda[feat].value_counts()/weather_eda[feat].count())
    print('---------------------------------')


# COMMAND ----------

# MAGIC %md
# MAGIC Following EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations Data

# COMMAND ----------

df_stations = spark.read.parquet('/mnt/mids-w261/datasets_final_project/stations_data/*').cache()
display(df_stations)

# COMMAND ----------

df_stations.count()

# COMMAND ----------

