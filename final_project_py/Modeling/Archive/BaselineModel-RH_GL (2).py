# Databricks notebook source
# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

main_data = joined_full_null_imputed_df.where(f.col('YEAR').between(2017, 2018)).cache() # GL NEW

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering
# MAGIC Flight arrived late is hypothesized to depart late for its next scheduled departure. Delays propagate.
# MAGIC `PRIOR_ARR_DEL15`: 
# MAGIC - 0 for prior arrival that is on-time or less than 15 minutes of delay
# MAGIC - 1 for prior arrival if more than 15 minutes of delay
# MAGIC - -1 for prior arrival past time of prediction (2 hours prior next scheduled departure time)
# MAGIC 
# MAGIC `PRIOR_DEP_DEL15`: 
# MAGIC - 0 for prior departure that is on-time or less than 15 minutes of delay
# MAGIC - 1 for prior departure if more than 15 minutes of delay
# MAGIC - -1 for prior departure within 15 minutes of time of prediction
# MAGIC   - Departure delay information will not be available
# MAGIC   - Highly unlikely scenario

# COMMAND ----------

def feature_engineering(main_data):
    # prior times of arrival/departure for same airplane
    window = Window.partitionBy('TAIL_NUM').orderBy('CRS_ARR_TIME_UTC')
    main_data = main_data.withColumn('PRIOR_ARR_TIME_UTC', 
                                     f.lag('ARR_TIME_UTC', 1, None).over(window))\
                         .withColumn('PRIOR_ARR_DEL15', 
                                     f.lag('ARR_DEL15', 1, None).over(window))\
                         .withColumn('TIME_OF_PREDICTION_UTC', 
                                     (f.unix_timestamp(f.col('CRS_DEP_TIME_UTC')) - 2*3600)\
                                     .cast('timestamp'))\
                         .withColumn('PRIOR_ARR_DEL15', f.when(f.col('PRIOR_ARR_TIME_UTC') <= 
                                                               f.col('TIME_OF_PREDICTION_UTC'), 
                                                               f.col('PRIOR_ARR_DEL15')).otherwise(-1))\
                         .withColumn('PRIOR_CRS_DEP_TIME_UTC', 
                                     f.lag('CRS_DEP_TIME_UTC', 1, None).over(window))\
                         .withColumn('PRIOR_DEP_DEL15',
                                     f.lag('DEP_DEL15', 1, None).over(window))\
                         .withColumn('PRIOR_DEP_DEL15', f.when((f.unix_timestamp(f.col('TIME_OF_PREDICTION_UTC')) - 
                                                                f.unix_timestamp(f.col('PRIOR_CRS_DEP_TIME_UTC'))) >= 15*60,
                                                                f.col('PRIOR_DEP_DEL15')).otherwise(-1))
    
    # airport stresses
    window_org = Window.partitionBy('ORIGIN_AIRPORT_ID').orderBy('TIME_OF_PREDICTION_UTC')\
                    .rangeBetween(-3*3600, -2*3600)
    window_dest = Window.partitionBy('DEST_AIRPORT_ID').orderBy('TIME_OF_PREDICTION_UTC')\
                    .rangeBetween(-3*3600, -2*3600)
    main_data = main_data.withColumn('TIME_OF_PREDICTION_UTC', 
                                     (f.unix_timestamp(f.col('CRS_DEP_TIME_UTC'))))\
                         .withColumn('PROB_DEP_DEL15_ORIGIN_PRIOR_1H',
                                     f.avg('DEP_DEL15').over(window_org))\
                         .withColumn('PROB_DEP_DEL15_DEST_PRIOR_1H',
                                     f.avg('DEP_DEL15').over(window_dest))
    
    # fillna
    main_data = main_data.fillna({'PROB_DEP_DEL15_ORIGIN_PRIOR_1H': 0,
        'PROB_DEP_DEL15_DEST_PRIOR_1H': 0})
    return main_data
    
# train_2017_null_imputed_df = feature_engineering(train_2017_null_imputed_df) # GL CO
# valid_2018_null_imputed_df = feature_engineering(valid_2018_null_imputed_df) # GL CO
main_data = feature_engineering(main_data).cache() # GL NEW
# main_data.select('FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'CRS_DEP_TIME_UTC', 'CRS_ARR_TIME_UTC', 'ARR_DEL15', 'PRIOR_ARR_DEL15', 'ARR_TIME_UTC', 'PRIOR_ARR_TIME_UTC', 'TIME_OF_PREDICTION_UTC').display()

# COMMAND ----------

main_data.count() # make the main data cache

# COMMAND ----------

# train_2017_df = main_data.where(f.col('YEAR') == 2017).cache()
# valid_2018_df = main_data.where(f.col('YEAR') == 2018).cache()
# test_2019_df = main_data.where(f.col('YEAR') == 2019).cache()

# COMMAND ----------

# train_2017_null_imputed_df.write.mode('overwrite').parquet(f'{blob_url}/model/train_2017_null_imputed_data.parquet')
# valid_2018_null_imputed_df.write.mode('overwrite').parquet(f'{blob_url}/model/valid_2018_null_imputed_data.parquet')
# test_2019_null_imputed_df.write.mode('overwrite').parquet(f'{blob_url}/model/test_2019_null_imputed_data.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impute null values for LR to run
# MAGIC - Use overall mean to fill missing weather information

# COMMAND ----------

def weather_impute_na(main_data, fill_null=False):

    weather_cols = [
        'ORIGIN_WEATHER_WND_DIRECTION_ANGLE-AVG',
        'ORIGIN_WEATHER_WND_SPEED_RATE-AVG',
        'ORIGIN_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
        'ORIGIN_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
        'ORIGIN_WEATHER_TMP_AIR_TEMP-AVG',
        'ORIGIN_WEATHER_DEW_POINT_TEMP-AVG',
        'ORIGIN_WEATHER_SLP_SEA_LEVEL_PRES-AVG',
        'DEST_WEATHER_WND_DIRECTION_ANGLE-AVG',
        'DEST_WEATHER_WND_SPEED_RATE-AVG',
        'DEST_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
        'DEST_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
        'DEST_WEATHER_TMP_AIR_TEMP-AVG',
        'DEST_WEATHER_DEW_POINT_TEMP-AVG',
        'DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG'
    ]

    if fill_null:
        for w in weather_cols:
            overall_mean = main_data.where(~f.col(w).isNull()).agg(f.avg(w)).first()[0]
            main_data = main_data.withColumn(w, f.when(f.col(w).isNull(), overall_mean).otherwise(f.col(w))).cache()
    
    return main_data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check for multicollinearity

# COMMAND ----------

# main_data_df = main_data.select(*TOT_FEATURES).cache().toPandas()
# sns.set(rc={'figure.figsize':(30, 20)})
# sns.heatmap(main_data_df.corr(), cmap='RdBu_r', annot=True)

# COMMAND ----------

# corr_matrix = main_data_df.corr().abs()
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
# to_drop

# COMMAND ----------

# main_data_reduced = train_2015_df.drop(*to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Round CRS_DEP_TIME up to the hour

# COMMAND ----------

def drop_minutes(df):
    # aggregate CRS time to hourly
    df = df.withColumn('CRS_DEP_TIME_HOUR', (df.CRS_DEP_TIME * 0.01).cast('int'))\
           .withColumn('CRS_ARR_TIME_HOUR', (df.CRS_ARR_TIME * 0.01).cast('int'))
    return df
main_data = drop_minutes(main_data) # GL NEW
# train_2017_null_imputed_df = drop_minutes(train_2017_null_imputed_df) # GL CO
# valid_2018_null_imputed_df = drop_minutes(valid_2018_null_imputed_df) # GL CO


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
    'DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG',
    
    'PROB_DEP_DEL15_ORIGIN_PRIOR_1H',
    'PROB_DEP_DEL15_DEST_PRIOR_1H',
]

CAT_FEATURES = [
#     'YEAR',
#     'QUARTER',
    'MONTH',
    'DAY_OF_MONTH',
    'DAY_OF_WEEK',
#     'FL_DATE',
    'CRS_DEP_TIME_UTC_HOUR',
    'CRS_DEP_TIME_HOUR',
  
    'PRIOR_ARR_DEL15',
    'PRIOR_DEP_DEL15',
    
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
len(TOT_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Split

# COMMAND ----------

# train_df = train_2017_null_imputed_df#main_data.where(f.col('MONTH').between(1, 9)).cache()
# valid_df = valid_2018_null_imputed_df#dev_2015_df.unionAll(test_2015_df)#.where(f.col('MONTH').between(10, 12)).cache()

# train_df = train_df[TOT_FEATURES]
# valid_df = valid_df[TOT_FEATURES]

# train_df = train_df.na.drop()
# valid_df = valid_df.na.drop()

# COMMAND ----------

main_data = main_data[TOT_FEATURES].cache()
main_data.count() # force to cache

# COMMAND ----------

# MAGIC %md
# MAGIC #### Address Class Imbalance

# COMMAND ----------

def class_balance(train_df, sample = 'undersample'):
    # train_df = main_data
    positive_df = train_df.filter(f.col('DEP_DEL15') == 1).cache()
    positive_count = positive_df.count()
    negative_df = train_df.filter(f.col('DEP_DEL15') == 0).cache()
    negative_count = negative_df.count()

#     sample = 'undersample'
    fraction = positive_count/negative_count

    if sample == 'undersample':
        train_df = negative_df.sample(withReplacement=False, 
                                      fraction=fraction, 
                                      seed = 1).unionAll(positive_df).cache()
    elif sample == 'oversample':
        train_df = positive_df.sample(withReplacement=True, 
                                      fraction=1/fraction, 
                                      seed = 1).unionAll(negative_df).cache()
    return train_df    
# main_data = train_df

# COMMAND ----------

# main_data.count() # force into cache

# COMMAND ----------

df_ordered = main_data.withColumn('SPLIT_ID', f.row_number().over(Window().orderBy(f.col('CRS_DEP_TIME_UTC_HOUR')))).cache()

# COMMAND ----------

# Removing UTC feat
CAT_FEATURES = [
#     'YEAR',
#     'QUARTER',
    'MONTH',
    'DAY_OF_MONTH',
    'DAY_OF_WEEK',
#     'FL_DATE',
#     'CRS_DEP_TIME_UTC_HOUR',
    'CRS_DEP_TIME_HOUR',
  
    'PRIOR_ARR_DEL15',
    'PRIOR_DEP_DEL15',
    
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

df_ordered = df_ordered.drop('CRS_DEP_TIME_UTC_HOUR')

# COMMAND ----------

df_ordered.display()

# COMMAND ----------

# write to parquet
# INTERIM_2017_18_ORDERED_PATH = blob_url + '/INTERIM/INTERIM_2017_18_ORDERED.parquet'
 
# df_ordered.write.mode('overwrite').parquet(INTERIM_2017_18_ORDERED_PATH)
# df_ordered = spark.read.parquet(INTERIM_2017_18_ORDERED_PATH).cache()

# COMMAND ----------

df_ordered.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### OneHotEncode

# COMMAND ----------

cat = []

for cat_feature in CAT_FEATURES:
    cat_SI = StringIndexer(inputCol = cat_feature, outputCol = cat_feature + '_SI', handleInvalid = 'keep')
    cat_OH = OneHotEncoder(inputCols = [cat_feature + '_SI'], outputCols = [cat_feature + '_OH'])
    cat.extend([cat_SI, cat_OH])
cat_VA = VectorAssembler(inputCols = [cat_feature + '_OH' for cat_feature in CAT_FEATURES], outputCol = 'CAT_FEATURES_VA')
cat.extend([cat_VA])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scale Continuous Features

# COMMAND ----------

scale_cont = True

if scale_cont:
    cont_va = VectorAssembler(inputCols = CONT_FEATURES, outputCol = 'CONT_FEATURES_VA_NONSCALE', handleInvalid = 'skip')
    scaler = RobustScaler(inputCol='CONT_FEATURES_VA_NONSCALE', 
                          outputCol = 'CONT_FEATURES_VA')
    cat.extend([cont_va, scaler])
else:
    cont_va = VectorAssembler(inputCols = CONT_FEATURES, outputCol = 'CONT_FEATURES_VA', handleInvalid = 'skip')
    cat.extend([cont_va])

# COMMAND ----------

all_features_va = VectorAssembler(inputCols = ['CAT_FEATURES_VA', 'CONT_FEATURES_VA'], outputCol = 'ALL_FEATURES_VA')
cat.append(all_features_va)

pipeline = Pipeline(stages = cat)

# model_pipeline = pipeline.fit(train_df)
# train_df = model_pipeline.transform(train_df)
# valid_df = model_pipeline.transform(valid_df)

# COMMAND ----------

def cv_folds(data = None, folds = 2, train_pct = .8, simple_splits = False):
    
    # Empty list to store dataframes for each fold
    train_dfs = []
    val_dfs = []
    
    # Create simple folds by full train year, val year
    if simple_splits:
        for year in [2015, 2016, 2017]:
            
            train_df = data.where(f.year(f.col('CRS_DEP_TIME_HOUR')) == year)
            val_df = data.where(f.year(f.col('CRS_DEP_TIME_HOUR')) == (year + 1))
            
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
            
            # Apply under/oversample to train data
            train_df = class_balance(train_df, 'undersample')
            
            
            # store data
            train_dfs.append(train_df)
            val_dfs.append(val_df)

    return train_dfs, val_dfs

# COMMAND ----------

# RH DISTILL


# COMMAND ----------

model_pipeline = pipeline.fit(df_ordered)

# COMMAND ----------

df_ordered_transf = model_pipeline.transform(df_ordered).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # Aditya take the below two cells (nevermind)

# COMMAND ----------

# write to parquet
# MODEL_TRANSFORMED_2017_18_CV_PATH_2 = blob_url + '/model_transformed_2017_18_CV_2.parquet'
 
# df_ordered_transf.write.parquet(MODEL_TRANSFORMED_2017_18_CV_PATH_2)
df = spark.read.parquet(MODEL_TRANSFORMED_2017_18_CV_PATH_2)

# COMMAND ----------

df = df.na.drop()
df = df.select('SPLIT_ID', 'DEP_DEL15', 'ALL_FEATURES_VA')

# COMMAND ----------

# train_dfs[0].agg({'SPLIT_ID': 'max'})

# COMMAND ----------

train_dfs, valid_dfs = cv_folds(data = df, folds = 2, train_pct = .8, simple_splits = False)

# COMMAND ----------

print(train_dfs[0].count())
print(train_dfs[1].count())
print(valid_dfs[0].count())
print(valid_dfs[1].count())

# COMMAND ----------

print(train_dfs[0].count())
print(train_dfs[1].count())
print(valid_dfs[0].count())
print(valid_dfs[1].count())

# COMMAND ----------

train_dfs[0].agg({'SPLIT_ID': 'max'}).display()
valid_dfs[0].agg({'SPLIT_ID': 'min'}).display()
valid_dfs[0].agg({'SPLIT_ID': 'max'}).display()
train_dfs[1].agg({'SPLIT_ID': 'max'}).display()
valid_dfs[1].agg({'SPLIT_ID': 'min'}).display()
valid_dfs[1].agg({'SPLIT_ID': 'max'}).display()

# COMMAND ----------

train_dfs[0].printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression

# COMMAND ----------

MAX_ITERS = [15, 20]
ALPHAS = [.1, .2]
LAMBDAS = [.01, .05]

param_list = [MAX_ITERS, ALPHAS, LAMBDAS]
names = ['MAX_ITERS', 'ALPHAS', 'LAMBDAS']

def gridsearch(full_data, k_folds, param_list, param_names):
    
    best_score = 0
    best_params = None
    
    # Assemble train/test from cv_folds
    train_dfs, valid_dfs = cv_folds(data = full_data, folds = k_folds, train_pct = .8, simple_splits = False)
    
    
    # Assemble Gridsearch
    params = list(itertools.product(*param_list))
    
    # Run Models in folds
    for param in params:
        
        # Instantiate a model with changing parameters as we search
        log_reg = LogisticRegression(labelCol = 'DEP_DEL15', 
                            featuresCol = 'ALL_FEATURES_VA', 
                            maxIter = params[0], 
                            elasticNetParam = params[1],
                            regParam = params[2])
        
        # Model through folds
        for fold in range(k_folds):
            log_reg_model = log_reg.fit(train_dfs[fold])
            preds_train = log_reg_model.transform(train_dfs[fold])
            preds_valid = log_reg_model.transform(valid_dfs[fold])

            train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
            train_metrics = MulticlassMetrics(train_rdd)
            valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
            valid_metrics = MulticlassMetrics(valid_rdd)

            # Score/compare model
            valid_score = valid_metrics.fMeasure(1.0, 0.5)
            if valid_score > best_score:
                best_score = valid_score
                best_params = param
            
            print('BASELINE LOGISTIC REGRESSION')
            print(param)
            print(' \t\tTrain Metrics \t Validation Metrics\n')
#             print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
#             print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
#             print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
            print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
#             print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
#             print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')
        
            
    return best_score, best_params

# COMMAND ----------



# COMMAND ----------

MAX_ITERS = [15, 20]
ALPHAS = [.1, .2]
LAMBDAS = [.01, .05]

grid = list(itertools.product(*[MAX_ITERS, ALPHAS, LAMBDAS]))

# COMMAND ----------

MAX_ITER = 15
ALPHA = 0.1
LAMBDA = 0.01

log_reg = LogisticRegression(labelCol = 'DEP_DEL15', 
                            featuresCol = 'ALL_FEATURES_VA', 
                            maxIter = MAX_ITER, 
                            elasticNetParam = ALPHA,
                            regParam = LAMBDA)

# log_reg_model = log_reg.fit(train_df)
# preds_train = log_reg_model.transform(train_df)
# preds_valid = log_reg_model.transform(valid_df)

# COMMAND ----------

# train_dfs_new = []
# valid_dfs_new = []
# train_dfs_new.append(train_dfs[0].select('DEP_DEL15', 'ALL_FEATURES_VA'))
# train_dfs_new.append(train_dfs[1].select('DEP_DEL15', 'ALL_FEATURES_VA'))
# valid_dfs_new.append(valid_dfs[0].select('DEP_DEL15', 'ALL_FEATURES_VA'))
# valid_dfs_new.append(valid_dfs[1].select('DEP_DEL15', 'ALL_FEATURES_VA'))


# COMMAND ----------

# train_dfs_new[0].count()

# COMMAND ----------

# log_reg_model = log_reg.fit(train_dfs[0])
# preds_train = log_reg_model.transform(train_dfs[0])
# preds_valid = log_reg_model.transform(valid_dfs[0])

# train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
# train_metrics = MulticlassMetrics(train_rdd)
# valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
# valid_metrics = MulticlassMetrics(valid_rdd)

# print('BASELINE LOGISTIC REGRESSION')
# print(' \t\tTrain Metrics \t Validation Metrics\n')
# print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
# print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
# print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
# print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
# print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
# print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

for fold in range(2):
    log_reg_model = log_reg.fit(train_dfs[fold])
    preds_train = log_reg_model.transform(train_dfs[fold])
    preds_valid = log_reg_model.transform(valid_dfs[fold])
    
    train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
    train_metrics = MulticlassMetrics(train_rdd)
    valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
    valid_metrics = MulticlassMetrics(valid_rdd)
    
    print('BASELINE LOGISTIC REGRESSION')
    print(' \t\tTrain Metrics \t Validation Metrics\n')
    print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
    print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
    print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
    print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
    print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
    print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')
    
    # if this model is best, save it - if not move on

# COMMAND ----------



# COMMAND ----------

# train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
# train_metrics = MulticlassMetrics(train_rdd)
# valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
# valid_metrics = MulticlassMetrics(valid_rdd)

# COMMAND ----------

# print('BASELINE LOGISTIC REGRESSION')
# print(' \t\tTrain Metrics \t Validation Metrics\n')
# print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
# print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
# print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
# print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
# print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
# print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forrest

# COMMAND ----------

# NUM_TREE = 128

# random_for = RandomForestClassifier(labelCol = 'DEP_DEL15', 
#                                  featuresCol = 'ALL_FEATURES_VA',
#                                  numTrees = NUM_TREE)

# random_for_model = random_for.fit(train_df)
# preds_train = random_for_model.transform(train_df)
# preds_valid = random_for_model.transform(valid_df)

# COMMAND ----------

# GL/RH
NUM_TREE = 128

random_for = RandomForestClassifier(labelCol = 'DEP_DEL15', 
                                 featuresCol = 'ALL_FEATURES_VA',
                                 numTrees = NUM_TREE)


    
for fold in range(2):
    random_for_model = random_for.fit(train_dfs[fold])
    preds_train = random_for_model.transform(train_dfs[fold])
    preds_valid = random_for_model.transform(valid_dfs[fold])
    
    train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
    train_metrics = MulticlassMetrics(train_rdd)
    valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
    valid_metrics = MulticlassMetrics(valid_rdd)

    print('BASELINE RANDOM FORREST')
    print(' \t\tTrain Metrics \t Validation Metrics\n')
    print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
    print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
    print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
    print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
    print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
    print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

# train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
# train_metrics = MulticlassMetrics(train_rdd)
# valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
# valid_metrics = MulticlassMetrics(valid_rdd)

# print('BASELINE RANDOM FORREST')
# print(' \t\tTrain Metrics \t Validation Metrics\n')
# print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
# print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
# print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
# print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
# print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
# print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBT

# COMMAND ----------

# MAX_ITER = 64
# MAX_DEPTH = 5
# MAX_BINS = 256
# STEP_SIZE = 0.1

# GBT = GBTClassifier(labelCol = 'DEP_DEL15', 
#                     featuresCol = 'ALL_FEATURES_VA', 
#                     maxIter = MAX_ITER,
#                     maxDepth = MAX_DEPTH,
#                     maxBins = MAX_BINS,
#                     stepSize = STEP_SIZE)

# GBT_model = GBT.fit(train_df)
# preds_train = GBT_model.transform(train_df)
# preds_valid = GBT_model.transform(valid_df)

# COMMAND ----------

# train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
# train_metrics = MulticlassMetrics(train_rdd)
# valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
# valid_metrics = MulticlassMetrics(valid_rdd)

# print('BASELINE GBT')
# print(' \t\tTrain Metrics \t Validation Metrics\n')
# print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
# print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
# print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
# print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
# print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
# print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

MAX_ITER = 64
MAX_DEPTH = 5
MAX_BINS = 256
STEP_SIZE = 0.1

GBT = GBTClassifier(labelCol = 'DEP_DEL15', 
                    featuresCol = 'ALL_FEATURES_VA', 
                    maxIter = MAX_ITER,
                    maxDepth = MAX_DEPTH,
                    maxBins = MAX_BINS,
                    stepSize = STEP_SIZE)

for fold in range(2):
    GBT_model = GBT.fit(train_dfs[fold])
    preds_train = GBT_model.transform(train_dfs[fold])
    preds_valid = GBT_model.transform(valid_dfs[fold])
    
    train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
    train_metrics = MulticlassMetrics(train_rdd)
    valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
    valid_metrics = MulticlassMetrics(valid_rdd)

    print('BASELINE GBT')
    print(' \t\tTrain Metrics \t Validation Metrics\n')
    print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
    print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
    print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
    print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
    print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
    print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')

# COMMAND ----------

