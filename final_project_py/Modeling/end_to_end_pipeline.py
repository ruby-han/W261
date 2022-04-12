# Databricks notebook source
# MAGIC %run /Users/rubyhan@berkeley.edu/team28/Final_Project/Imports

# COMMAND ----------

class DataSplit():
    "Class to split data. Used as a parent class for Models"
    
    def __init__(self,df, train_percent=0.8, timestamp_col='CRS_DEP_TIME_UTC', folds = 1, simple_splits = False, blind_test = None):
        self.df = df
        self.train_percent = train_percent
        self.test_percent = 1 - train_percent
        self.col_name = timestamp_col
        self.folds = folds
        self.simple_splits = simple_splits
        self.train_df = []
        self.test_df = []
        if blind_test is not None:
            self.test_df = [blind_test]
            self.train_df = [DataSplit.class_balance(df, 'undersample')]
    
    def split_on_timestamp(self):
        
        min_ts = self.df.select(min(self.col_name).alias('min_ts')).take(1)[0]['min_ts']
        max_ts = self.df.select(max(self.col_name).alias('max_ts')).take(1)[0]['max_ts']
        time_range = (max_ts - min_ts).days
        train_threshold_days = min_ts + dt.timedelta(days=int(self.train_percent*time_range))
        df_train = self.df.filter(col(self.col_name) < train_threshold_days)
        df_test = self.df.filter(col(self.col_name) >= train_threshold_days)
        self.train_df = [df_train]
        self.test_df = [df_test]

    @staticmethod
    def class_balance(train_df, sample = 'undersample'):
        positive_df = train_df.filter(f.col('DEP_DEL15') == 1).cache()
        positive_count = positive_df.count()
        negative_df = train_df.filter(f.col('DEP_DEL15') == 0).cache()
        negative_count = negative_df.count()

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

    def cv_folds(self):
        
        # Create simple folds by full train year, val year
        if self.simple_splits is True:
            for year in [2015, 2016, 2017]:

                train_df = self.df.where(f.col('CRS_DEP_TIME_UTC_HOUR') == year)
                val_df = self.df.where(f.col('CRS_DEP_TIME_UTC_HOUR') == (year + 1))

                self.train_df.append(train_df)
                self.test_df.append(val_df)
            
        # Create folds by number of folds, train/test split %
        else:
            data_size = self.df.count()

            for k in range(self.folds): 

                # Calculate total size of each fold
                fold_size = data_size/self.folds

                # Find range of `SPLIT_ID` for train and val
                train_ids = ((fold_size * k) + 0, 
                             int(fold_size * k + fold_size*self.train_percent))

                val_ids = (train_ids[1] + 1, 
                           fold_size * k + fold_size)            

                # Split the data
                train_df = self.df.where(f.col('SPLIT_ID').between(train_ids[0], train_ids[1]))
                val_df = self.df.where(f.col('SPLIT_ID').between(val_ids[0], val_ids[1]))

                # store data
                train_df = DataSplit.class_balance(train_df, 'undersample')
                self.train_df.append(train_df)
                self.test_df.append(val_df)

#         return train_dfs, val_dfs

class Model(DataSplit):
    
#     def __init__(self,train_df, test_df, label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'], **kwargs):
    def __init__(self, df,label_col=['DEP_DEL15'],feature_col=['ALL_FEATURES_VA'], train_percent=0.8, timestamp_col='CRS_DEP_TIME_UTC', folds=1, simple_splits=False, blind_test=None):
        
        super().__init__(df, train_percent, timestamp_col, folds, simple_splits, blind_test)
        self.label_col = label_col
        self.feature_col = feature_col
        self.model = None
        self.best_score = 0
        self.best_params = None
        self.best_preds_test = None
    
    def get_logistic_regression(self, **kwargs):
        model_args = kwargs
        lr = LogisticRegression(labelCol = self.label_col[0],
                                  featuresCol=self.feature_col[0],
                                  elasticNetParam = model_args.get('elasticNetParam',1),
                                  standardization = model_args.get('standardization',True),
                                  maxIter=model_args.get('maxIter',10), 
                                  regParam=model_args.get('regParam',0.001))
        
        for index in range(len(self.train_df)):
            print(f'\n{index}-Fold:\n')
            self.model = lr.fit(self.train_df[index])
            self.model.setThreshold(model_args.get('threshold',0.5))
            self.get_predictions(index,model_args)
    
    def get_random_forest(self,**kwargs):
        model_args = kwargs
        rf = RandomForestClassifier(labelCol = self.label_col[0],
                                    featuresCol=self.feature_col[0],
                                    numTrees=model_args.get('numTrees',1),
                                    maxDepth=model_args.get('maxDepth',5),
                                    maxBins=model_args.get('maxBins',32))
         
        for index in range(len(self.train_df)):
            print(f'\n{index}-Fold:\n')
            self.model = rf.fit(self.train_df[index])
            self.get_predictions(index,model_args)
            
    def get_GBT(self,**kwargs):
        model_args = kwargs
        GBT = GBTClassifier(labelCol = self.label_col[0],
                            featuresCol=self.feature_col[0],
                            maxIter=model_args.get('maxIter',1),
                            maxDepth=model_args.get('maxDepth',5),
                            maxBins=model_args.get('maxBins',32),
                            stepSize=model_args.get('stepSize',0.1))
         
        for index in range(len(self.train_df)):
            print(f'\n{index}-Fold:\n')
            self.model = GBT.fit(self.train_df[index])
            self.model.setThresholds([1-model_args.get('threshold',0.5), model_args.get('threshold',0.5)])
            self.get_predictions(index,model_args) 

    
    def get_predictions(self,index,model_args):
        results = self.model.transform(self.test_df[index])
        preds_train = self.model.transform(self.train_df[index])
        self.get_metrics(preds_train,results,model_args)
  
    def get_confusion_matrix(self):
        
        pandas_df = self.best_preds_test.select([self.label_col[0],'prediction']).toPandas()
        cf_matrix = confusion_matrix(y_true=pandas_df[self.label_col], y_pred=pandas_df['prediction'],normalize='true')
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        plt.show(sns)
    
    def get_metrics(self,preds_train, preds_test,model_args):
        
        train_rdd = preds_train.select(['prediction', 'DEP_DEL15']).rdd
        train_metrics = MulticlassMetrics(train_rdd)
        valid_rdd = preds_test.select(['prediction', 'DEP_DEL15']).rdd
        valid_metrics = MulticlassMetrics(valid_rdd)
        valid_score = valid_metrics.fMeasure(1.0, 0.5)
        if valid_score > self.best_score:
            self.best_score = valid_score
            self.best_params = model_args
            self.best_preds_test = preds_test
        print(' \t\tTrain Metrics \t Validation Metrics\n')
        print(f'Recall: \t\t{train_metrics.recall(label=1):.3f} \t\t {valid_metrics.recall(label=1):.3f}')
        print(f'Precision: \t\t{train_metrics.precision(1):.3f} \t\t {valid_metrics.precision(1):.3f}')
        print(f'Accuracy: \t\t{train_metrics.accuracy:.3f} \t\t {valid_metrics.accuracy:.3f}')
        print(f'F0.5 score: \t\t{train_metrics.fMeasure(1.0, 0.5):.3f} \t\t {valid_metrics.fMeasure(1.0, 0.5):.3f}')
        print(f'F2 score: \t\t{train_metrics.fMeasure(1.0, 2.0):.3f} \t\t {valid_metrics.fMeasure(1.0, 2.0):.3f}')
        print(f'F1 score: \t\t{train_metrics.fMeasure(1.0):.3f} \t\t {valid_metrics.fMeasure(1.0):.3f}')
        
    def feature_rank(self):
        # Only works for tree based models
    # plot feature importance from tree models
        feature =  self.best_preds_test.schema['ALL_FEATURES_VA'].metadata['ml_attr']['attrs']
        dict_feature={}
        for key in feature.keys():
            for i in range(len(feature[key])):
                dict_feature[feature[key][i]['idx']]= feature[key][i]['name']
        features_df = pd.DataFrame(list(dict_feature.items()), columns = ['index', 'name'])
        features_rank = self.model.featureImportances
        features_rank_arr = pd.DataFrame(features_rank.toArray())
        features_rank_arr.rename(columns={0:'score'}, inplace=True)

        top_20 = features_rank_arr.sort_values('score', ascending = 
                                               False).head(20).reset_index()
        df = pd.merge(features_df, top_20, on = 'index').sort_values('score', ascending=False)

        sns.set(font_scale=1, style='whitegrid');
        plt.subplots(figsize=(20, 15))
        ax = sns.barplot(x = 'score', y = 'name', data = df)
        ax.set_xlabel('Feature Importance Score')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance Rank');
        plt.show(sns)

    def tune_threshold(self):
        valid_df = self.best_preds_test
        pr_list = []
        element1 = udf(lambda item: float(item[1]), FloatType())
        prediction = valid_df.withColumn('prediction_prob', element1('probability'))
        threshold_range = np.arange(start = 0.1, stop = 1.1, step = 0.1)
        x = []
        for i in range(1, 11):
            x.append(f'x{i}')
        i = 0
        for threshold in threshold_range:
            prediction = prediction.withColumn(x[i], f.when(
                prediction['prediction_prob'].cast(DoubleType()) >= threshold,
                1.0).otherwise(0.0).cast(DoubleType()))
            i += 1    
        for i in range(len(threshold_range) - 1):
            preds_rdd = prediction.select([x[i], 'DEP_DEL15']).rdd
            preds_metrics = MulticlassMetrics(preds_rdd)
            precision = preds_metrics.precision(1)
            recall = preds_metrics.recall(label=1)
            f0_5 = preds_metrics.fMeasure(1.0, 0.5)
            pr_list.append((threshold_range[i], precision, recall, f0_5))
        return pd.DataFrame(pr_list).rename(columns = {0: 'Threshold', 1: 'Precision', 
                                                       2: 'Recall', 3: 'F0_5-Score'})

    def threshold_plot(self, preds_valid_PR):
        # plot precision-recall-f0.5 curve to tune threshold
        sns.set(font_scale=1, style='whitegrid')
        sns.lineplot(x='Threshold',y='Recall',data=preds_valid_PR,label='Recall')
        sns.lineplot(x='Threshold',y='Precision',data=preds_valid_PR,label='Precision')
        sns.lineplot(x='Threshold',y='F0_5-Score',data=preds_valid_PR,label='F0.5 Score')
        plt.vlines(0.5, 0, 1, color='red')  
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Tuning')
        plt.legend()
        plt.show(sns);

# COMMAND ----------

def gridsearch(full_data, k_folds, param_list, param_names, random_shuffle_top_N=10,model_type='lr', threshold_plot=False, confusion_matrix=False, feature_rank=False, blind_test=None):
       
    # Assemble train/test from cv_folds
    model = Model(df=full_data, folds = k_folds, train_percent = .8, simple_splits = False, blind_test=blind_test)
    if blind_test is None:
        model.cv_folds()
      
    # Assemble Gridsearch
    params = list(itertools.product(*param_list))
    random.shuffle(params)
    # Run Models in folds
    for param in params[:random_shuffle_top_N]:
        
        print(f"\nLooping through param:{list(zip(param_names,param))}")
        if model_type == 'lr':
            model.get_logistic_regression(**{'maxIter':param[0],'elasticNetParam':param[1],'regParam':param[2], 'threshold':param[3]})
        if model_type == 'rf':
            model.get_random_forest(**{'numTrees':param[0],'maxDepth':param[1],'maxBins':param[2]})
        if model_type == 'gbt':
            model.get_GBT(**{'maxIter':param[0],'maxDepth':param[1],'maxBins':param[2],'stepSize':param[3], 'threshold':param[4]})
    
    if k_folds > 1:
        print(f'\nBest Valid f0.5 Score: {model.best_score:.3f}\nBest Parameters: {model.best_params}')
    
    if threshold_plot:
        preds_valid_PR = model.tune_threshold() # TO DO: chagne results to preds_valid
        model.threshold_plot(preds_valid_PR)
    
    if feature_rank:
        model.feature_rank()
        
    if confusion_matrix:
        model.get_confusion_matrix()