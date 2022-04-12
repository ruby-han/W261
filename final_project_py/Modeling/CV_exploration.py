# Databricks notebook source
# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

LABEL = ['DEP_DEL15']

CONT_FEATURES = [
    'CRS_ELAPSED_TIME',
    'AIRPORT_LAT_ORIGIN',
    'AIRPORT_LONG_ORIGIN',
    'AIRPORT_LAT_DEST',
    'AIRPORT_LONG_DEST',
    
    'ORIGIN_WEATHER_ELEVATION',
    'ORIGIN_WEATHER_WND_DIRECTION_ANGLE-AVG',
    'ORIGIN_WEATHER_WND_SPEED_RATE-AVG',
    'ORIGIN_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
    'ORIGIN_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
#     'ORIGIN_WEATHER_TMP_AIR_TEMP-AVG',
    'ORIGIN_WEATHER_DEW_POINT_TEMP-AVG',
    'ORIGIN_WEATHER_SLP_SEA_LEVEL_PRES-AVG',
    
    'DEST_WEATHER_ELEVATION',
    'DEST_WEATHER_WND_DIRECTION_ANGLE-AVG',
    'DEST_WEATHER_WND_SPEED_RATE-AVG',
    'DEST_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
    'DEST_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
#     'DEST_WEATHER_TMP_AIR_TEMP-AVG',
    'DEST_WEATHER_DEW_POINT_TEMP-AVG',
    'DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG'
]

CAT_FEATURES = [
#     'YEAR',
#     'QUARTER',
    'MONTH',
    'DAY_OF_MONTH',
    'DAY_OF_WEEK',
#     'FL_DATE',
    'CRS_DEP_TIME_UTC_HOUR',
#     'CRS_DEP_TIME_HOUR',
  
#     'PRIOR_ARR_DEL15',
#     'PRIOR_DEP_DEL15',
    
    'OP_UNIQUE_CARRIER',
    'TAIL_NUM',
    'ORIGIN_AIRPORT_ID',
    'ORIGIN',
    'DEST_AIRPORT_ID',
    'DEST',
    'DEP_TIME_BLK',
#     'ARR_TIME_BLK',
#     'ID',
    
]

TOT_FEATURES = LABEL + CONT_FEATURES + CAT_FEATURES
# len(TOT_FEATURES)

# COMMAND ----------

# Ease
df = joined_full_processed_df[TOT_FEATURES].cache()

# COMMAND ----------

df.count()

# COMMAND ----------

df.select('CRS_DEP_TIME_UTC_HOUR').display().head(10)

# COMMAND ----------

# orderBy Gaurantees total sort, whereas sort is within partition https://towardsdatascience.com/sort-vs-orderby-in-spark-8a912475390
df_ordered = df.withColumn('SPLIT_ID', f.row_number().over(Window().orderBy(f.col('CRS_DEP_TIME_UTC_HOUR')))).cache()

# COMMAND ----------

df_ordered.select('CRS_DEP_TIME_UTC_HOUR', 'SPLIT_ID').display()

# COMMAND ----------

df_ordered.count()

# COMMAND ----------

def cv_folds(data = None, folds = 1, train_pct = .8, simple_splits = False):
    
    # Empty list to store dataframes for each fold
    train_dfs = []
    val_dfs = []
    
    # Create simple folds by full train year, val year
    if simple_splits:
        for year in [2015, 2016, 2017]:
            
            train_df = data.where(f.col('CRS_DEP_TIME_UTC_HOUR') == year)
            val_df = data.where(f.col('CRS_DEP_TIME_UTC_HOUR') == (year + 1))
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
    
    # Create folds by number of folds, train/test split %
    else:
        data_size = data.count()
        
        for k in range(folds): 

            # Calculate total size of each fold
            fold_size = data_size/folds

            # Find range of `SPLIT_ID` for train and val
            train_ids = ((fold_size * k) + 0, 
                         int(fold_size * k + fold_size*.8))

            val_ids = (train_ids[1] + 1, 
                       fold_size * k + fold_size)            

            # Split the data
            train_df = data.where(f.col('SPLIT_ID').between(train_ids[0], train_ids[1]))
            val_df = data.where(f.col('SPLIT_ID').between(val_ids[0], val_ids[1]))

            # store data
            train_dfs.append(train_df)
            val_dfs.append(val_df)

    return train_dfs, val_dfs




# COMMAND ----------

sample = df_ordered.filter(f.col('SPLIT_ID') < 200000).cache()


# COMMAND ----------

sample.printSchema()

# COMMAND ----------

simple = False

def test_f(simple = True):
    if not simple: 
        print('false')
    else:
        print('true')
        
test_f(simple)

# COMMAND ----------

sample.count()

# COMMAND ----------

tr, va = cv_folds(sample, 2, .8)

# COMMAND ----------

va[1].count()

# COMMAND ----------

tr2[1].count()

# COMMAND ----------

val_df_1.agg({'SPLIT_ID': 'max', 'SPLIT_ID': 'min'}).show()
train_df_1.agg({'SPLIT_ID': 'max', 'SPLIT_ID': 'min'}).show()

# COMMAND ----------

for i in range(2):
    print(i)
    tr[i].agg({'SPLIT_ID': 'min'}).show()
    tr[i].agg({'SPLIT_ID': 'max'}).show()
    va[i].agg({'SPLIT_ID': 'min'}).show()
    va[i].agg({'SPLIT_ID': 'max'}).show()

# COMMAND ----------

data_size = sample.count()
# fold_size = data_size/folds

# train_ids = (0, int(fold_size*.8))
# val_ids = (train_ids[1] + 1, fold_size)



# train_df = data.where(f.col('SPLIT_ID') < train_ids[1])
data_size

# COMMAND ----------

sample.display()

# COMMAND ----------

sample.withColumn('Year', f.year(f.col("CRS_DEP_TIME_UTC_HOUR"))).select(["CRS_DEP_TIME_UTC_HOUR", 'Year']).display()

# COMMAND ----------

sample.where(f.year(f.col('CRS_DEP_TIME_UTC_HOUR')) == 2015).display()

# COMMAND ----------

