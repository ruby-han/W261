# Databricks notebook source
# MAGIC %md
# MAGIC # Grid Search For Optimal Hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run our Imports Block

# COMMAND ----------

# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

# # Simplify
df = model_transformed_train_2018_df.cache()
df = df.na.drop()
df = df.select('SPLIT_ID', 'DEP_DEL15', 'ALL_FEATURES_VA')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pulling in CV function (this is already in OOP framework notebook)

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

# MAGIC %md
# MAGIC ### Pulling in `class_balance` function @abajaj NEEDS to be ported into OOP framework

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gridsearch Function - NEEDS to be ported to OOP Framework (@abajaj)

# COMMAND ----------

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
                            maxIter = param[0], 
                            elasticNetParam = param[1],
                            regParam = param[2])
        
        
        GBT = GBTClassifier(labelCol = 'DEP_DEL15', 
                    featuresCol = 'ALL_FEATURES_VA', 
                    maxIter = param[0],
                    maxDepth = param[1],
                    maxBins = param[2],
                    stepSize = param[3])
        
        # Model through folds 
        for fold in range(k_folds):
            
            # @abajaj thinking lines 27-34 will be covered by a `model` class
#             log_reg_model = log_reg.fit(train_dfs[fold])
#             preds_train = log_reg_model.transform(train_dfs[fold])
#             preds_valid = log_reg_model.transform(valid_dfs[fold])
            
            GBT_model = GBT.fit(train_dfs[fold])
            preds_train = GBT_model.transform(train_dfs[fold])
            preds_valid = GBT_model.transform(valid_dfs[fold])
            
            train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
            train_metrics = MulticlassMetrics(train_rdd)
            valid_rdd = preds_valid.select(['prediction', 'DEP_DEL15']).rdd
            valid_metrics = MulticlassMetrics(valid_rdd)

            # Score/compare model
            valid_score = valid_metrics.fMeasure(1.0, 0.5)
            if valid_score > best_score:
                best_score = valid_score
                best_params = param
            
            print(f'Results for gridsearch, fold {fold}:')
            print(f'{param_names[0]}: {param[0]} | {param_names[1]}: {param[1]} | {param_names[2]}: {param[2]} | {param_names[3]}: {param[3]}')
            print(' \t\tTrain Metrics \t Validation Metrics\n')
#             print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
#             print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
#             print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
            print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_score:.3f}\n')
#             print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
#             print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')
           
    return best_score, best_params

# COMMAND ----------

# MAX_ITERS = [64, 256]
# ALPHAS = [0.2, 0.3]
# LAMBDAS = [.01, 0.001]

# param_list = [MAX_ITERS, ALPHAS, LAMBDAS]
# names = ['MAX_ITERS', 'ALPHAS', 'LAMBDAS']

MAX_ITERS = [6, 10, 15]
MAX_DEPTH = [5, 10]
MAX_BINS = [128, 256, 280]
STEP_SIZE = [0.01, 0.1, 0.2]

param_list = [MAX_ITERS, MAX_DEPTH, MAX_BINS, STEP_SIZE]
names = ['MAX_ITERS', 'MAX_DEPTH', 'MAX_BINS', 'STEP_SIZE']

score, b_params = gridsearch(df, 3, param_list, names)

# COMMAND ----------

print(score)
print(b_params)

# COMMAND ----------

