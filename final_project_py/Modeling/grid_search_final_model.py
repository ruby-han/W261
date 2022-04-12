# Databricks notebook source
# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Modeling/end_to_end_pipeline

# COMMAND ----------

df = model_transformed_train_2015_18_new_pagerank_noncontscale_df.select('DEP_DEL15', 'ALL_FEATURES_VA')
df = df.na.drop()

# COMMAND ----------

# Log Reg
# MAX_ITERS = [16, 32, 64]
# ALPHAS = [.1, .2, .3] 
# LAMBDAS = [.01, .05, .1]

# param_list = [MAX_ITERS, ALPHAS, LAMBDAS]
# names = ['MAX_ITERS', 'ALPHAS', 'LAMBDAS']

# Random Forest
# NUM_TREES = [128,256,512]
# MAX_DEPTH = [10,20,40,80]
# MAX_BINS=[128,256,512]
# param_list = [NUM_TREES,MAX_DEPTH,MAX_BINS]
# names = ['NUM_TREES','MAX_DEPTH','MAX_BINS']

# GBT
MAX_ITERS = [6]
MAX_DEPTH = [6, 10]
MAX_BINS = [256, 280]
STEP_SIZE = [0.1, 0.2]

param_list = [MAX_ITERS, MAX_DEPTH, MAX_BINS, STEP_SIZE]
names = ['MAX_ITERS', 'MAX_DEPTH', 'MAX_BINS', 'STEP_SIZE']

gridsearch(df, k_folds=3, param_list, names, random_shuffle_top_N=5, model_type='gbt')