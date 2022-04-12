# Databricks notebook source
# MAGIC %md
# MAGIC # Phases Summaries
# MAGIC > *A brief summary of your progress by addressing the requirements for each phase. These should be concise, one or two paragraphs. There is a character limit in the form of 1000 for these*
# MAGIC 
# MAGIC ### Phase I (Due 3/13 @ Midnight)
# MAGIC Our initial EDA tasks were focused on understanding the data and features, exploring null values, and determining metrics for evaluation of our future model. Early in the process we were able to identify that our data is heavily imbalanced - with only 20% of our data classified as a flight delay. Second, we narrowed down the list of features based on proportion of null values. We dropped roughly half of the features in our `airlines` dataset, and 90% of the features in `weather`.
# MAGIC 
# MAGIC We also focused on how we would augment our flight data to increase forecast performance. To do this, include two different auxiliary tables - `stations` which is provided, and an airport to lat/long mapping, pulled from the web. A series of three joins described below will be necessary to create our full dataset. 
# MAGIC 
# MAGIC Finally, we decided that the metric best suited to help airlines reduce costs would be precision. This metric directly measures the share of false positives in relation to total positives, and gives us a mechanism to measure how often we incorrectly predict a delay when a delay does not actually occur. We believe these false positives to be the most costly mistakes. (*~972 chars not including spaces*)
# MAGIC 
# MAGIC ### Phase II (Due 3/20 @ midnight)
# MAGIC 
# MAGIC Our EDA continuation was heavily focused on understanding null values and treating them. Following removal of features with 70% or greater null values, we treated missing data by applying a 7-day look-back window. In addition to imputation of null values, we ran a number of transformations like time-stamp conversions, lags, etc in preparation for joining of weather data to airline data. These treatments allowed a reducton of null-values from ~30%, to less than 2.5%. 
# MAGIC 
# MAGIC Following Null value treatment and other pipeline transformations, we left joined weather to airlines, using stations as a mapper. Instead of a custom distance metric using lat/long, we mapped the shortest ICAO-to-station_ID, and pulled weather for these stations. 
# MAGIC 
# MAGIC Finally, we assembled a baseline model using logistic regression, and the 26 categorical and continuous variables we believed held the highest predictive power. Without full treatment of class imbalance, our model was unable to predict well, and produed precision scores in the 20%s. Next stages will focus on feature engineering, over-/under-sampling, sliding window CV, and other techniques to increase model power.  
# MAGIC 
# MAGIC Phase II updates included in **EDA & Discussion of Challenges** in addition to Phase I, and **Feature Engineering and Transformations**
# MAGIC 
# MAGIC 
# MAGIC ### Phase III (Due 4/3 @ midnight)
# MAGIC 
# MAGIC The first portion of this phase focused on implementing our join infrastructure from phase II to the full dataset. Our join took roughly 5 minutes with the help of caching, treatment of data before joining, and reature reductino. 
# MAGIC 
# MAGIC The second portion of this phase was focused on building a cross-validation and gridsearch framework that would allow for iteration over our master data set, in order to search for best combinations of hyperparameters. 
# MAGIC 
# MAGIC In parallel, other group members focused on creating three new features related to Airport stress, centrality, and prior flight delays. 
# MAGIC 
# MAGIC Finally, we iterated over many combinations of features, train/test sizes, and hyperparameters to reach a phase III F0.5 score of .47.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 3 Gap Analysis
# MAGIC 
# MAGIC At the time of writing (9:58p PT), our F0.5 performance relative to other groups scoring F0.5 put us at the middle of the pack, in the high .40s. Three other groups outperformed us with scores upwards of 70%. We believe we can improve our model in three ways as we move into Phase IV, in order of decreasing effectiveness: 
# MAGIC 
# MAGIC 1. Expanding on feature engineering with additional domain knowledge - Will raise the ceiling on F0.5
# MAGIC 2. Tuning log regression threshold
# MAGIC 3. Principle Component Analysis - Feature reduction/selection to attempt reduction in overfit
# MAGIC 
# MAGIC Our full join pipeline performed toward the top of all groups at less than 10 minutes, which we attribute to heavy use of caching as well as purposefull cleaning/feature reduction before join. 

# COMMAND ----------

# displayHTML("<img src ='files/shared_uploads/gerrit.lensink@berkeley.edu/Screen_Shot_2022_04_03_at_6_46_37_PM.png'>")
# displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/eda.png'>")
# displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/join.png'>")
# displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/phase_3_results.png'>")
# displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/cv_example.png'>")

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Formulation
# MAGIC 
# MAGIC Flight delays are an incredibly costly consequence in the airline industry - affecting airlines with lost revenue and lost demand, as well as costing passengers time and money. The FAA estimates that in 2019, flight delays caused losses of roughly 33 billion dollars [1], where airlines absorb 25% of the overall direct losses, and passengers receive over 50% of the losses. These delays cause direct impacts, like lost revenue for airlines or wasted time for customers, as well as indirect impacts like built-in planning buffers. These buffers then result in lower plane and resource utilization leading to increased ticket prices and other secondary costs, like lost "good will" and competitiveness with other airlines. 
# MAGIC 
# MAGIC Because of these high costs, airlines are motivated to predict delays as soon as possible to save losses both for their business and their customers. Current state of the art models, like neural nets are able to predict with precison in the low to high 90%s [2]
# MAGIC 
# MAGIC 
# MAGIC The purpose of our project is to provide a model that predicts flight delays, specifically tailored for use by airline companies. We define a delay as any departure that occurs 15 minutes or greater than the scheduled departure time. We will use precision to score the efficacy of our model, and wish to achieve precision scores of 70% or greater. Further discussion of our metric is included in Phase 1.3 below. 
# MAGIC 
# MAGIC [1] https://www.faa.gov/data_research/aviation_data_statistics/media/cost_delay_estimates.pdf
# MAGIC 
# MAGIC [2] https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00380-z

# COMMAND ----------

# MAGIC %md
# MAGIC # I. EDA & Discussion of Challenges
# MAGIC 
# MAGIC External Notebook links:
# MAGIC * [Initial EDA on Airline, weather, and stations datasets](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885897629/command/1038316885897636)
# MAGIC * [Further EDA on weather dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885900748/command/1038316885900753)
# MAGIC * [Processing/Transforming Airline Data](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2866290024316332/command/2866290024316333)
# MAGIC * [Processing/Transforming Weather Data](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2866290024316479/command/2866290024316486)
# MAGIC * [Processing/Transforming Station Data](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2866290024316534/command/2866290024316535)
# MAGIC * [Joins Notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3816960191355236/command/3816960191355237)
# MAGIC 
# MAGIC ### PHASE I EDA (with some updates from Phase II): 
# MAGIC 
# MAGIC **Airline Data**
# MAGIC This dataset consists of 109 different columns describing airline data, with two different subsets - Q1 of 2015, or Q1 + Q2 of 2015. For our initial EDA we focused on the 6 month dataset, assuming it is fairly representative of the overall dataset. This 6-month subset had 337k different observations. Our target variable is `DEP_DEL15`, which is a binary variable that is '1' when the given flight is delayed by 15 mins or more, and '0' when there is no delay or the delay is less than 15 minutes. Across the entire sample, we've observed an average delay of 14 minutes, with 36 minutes standard deviation. Our group will pay special attention to the fact that the dataset is highly imbalanced - with only ~21% of the flight records including a delay of 15 or more minutes. 
# MAGIC 
# MAGIC During data integrity checks, ~48 columns were identified as consisting of 90% or greater null values. These columns were dropped as they contained very little userful information and treatment of missing values would be innacurate. Additionally, we are able to drop columns related to arrival information, which further allowed us to reduce the size of our dataset. These columns contained data that would be considered leakage, i.e. at departure time - 2 hrs, we would not know about different arrival metrics for the same leg of the flight - therefore they must be removed. 
# MAGIC 
# MAGIC In addition to checks for null values - we also focused on identification of duplicate data. This was accomplished by creating a new unique_id column, which is a concatenation of `FL_DATE`, `TAIL_NUM`, `ORIGIN`, and `CRS_DEP_TIME`. Duplicates identified are commonly related to cancelled flights that are later replaced by flights to a different destination, with same `FL_DATE`, `TAIL_NUM`, and `ORIGIN`. Cancelled flights were dropped, and replacement flights were retained. 
# MAGIC 
# MAGIC **Weather Data**: 
# MAGIC 
# MAGIC The weather dataset (6 months) consists of 29,823,926 (29.8 million) rows and 177 columns. Across the 177 columns, 156 columns consist of 60% or greater null values. This high proportion of nulls motivates removal of these columns - as treatment of this many nulls would be inaccurate. Additional decoding was also included, with features like `WND`, `CIG`, `VIS`, `TMP`, `DEW`, and `SLP` were composite - and were split into relevant columns rather than retaining the concatenated values. 
# MAGIC 
# MAGIC During initial EDA, any columns with more than 70% null values/blank strings/`nan` were dropped. The columns of interest which will be used for analysis are included in the **Features of Interest** Section below.
# MAGIC 
# MAGIC **Stations Data**: 
# MAGIC 
# MAGIC The stations dataset was used to connect the closest weather station to each airport in the dataset. This dataset is a flat representation of a distance matrix, mapping each weather station to all other weather stations by distance. Each of the 5 million observation has non-null values across the 12 features, so no treatment of null values was required. Our Phase II join (discussed in detail below) uses the station dataset to map ICAO code in `Airline` to `Station_ID` in `Weather`. 
# MAGIC 
# MAGIC #### Outcome of interest
# MAGIC 
# MAGIC Our model will predict whether a flight is expected to be delayed by more than 15 minutes up to two hours before the scheduled departure time. Specifically, our target variable is `DEP_DEL15`, where '1' represents a flight delay of 15 minutes or greater, and 0 represents an on-time departure (or an 'insignificant' delay of less than 15 minutes). 
# MAGIC 
# MAGIC Because our model is built to reduce costs specifically for the airline, we focus on minimzing the rate at which we predict false positives, which is defined as a prediction of a delay, when the flight will not actually require a delay. We have chosen to focus on this specifically because as a false positive would trigger many costs for the airline like lost revenue due to empty seats, flight changes by customers, etc. We define precision as \\( \frac {FP}{FP + TP} \\), where \\(FP\\) = False Positive and \\(TP \\) = True Positive
# MAGIC 
# MAGIC #### EDA Summary
# MAGIC 
# MAGIC The two biggest challenges our EDA presented were class impbalance, and high proportion of features with Null Data. Before joining or training, it was important that we properly accounted for each of these issues. Techniques employed to address these issues are covered throughout the remainder of Section I and Section II. 

# COMMAND ----------

displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/eda.png'>")

# COMMAND ----------

# MAGIC %md
# MAGIC ### STAGE II EDA (with some updates to Phase I)
# MAGIC 
# MAGIC #### Data Ingestion
# MAGIC 
# MAGIC Our group has chosen to heavily leverage the use of parquet files, both for reading in data, as well as saving to blob. In general, parquet's column-wise format is more efficient both from a storage perspective as well as read speed compared to CSV or txt format [9]. We have realized significant effiency from caching large parquet files, as well as spark Dataframes. These cachings allow us to interact with data quicker, and also enable quicker joins. Storing commonly-used tables in the cloud allows for quicker start-up during node start, and efficient sharing of files across notebooks. 
# MAGIC 
# MAGIC During analysis, data is represented primarily as spark dataframes, with minimal usage of pandas - only for plotting. This method
# MAGIC 
# MAGIC #### Joins
# MAGIC 
# MAGIC Creation of our final dataset required heavy transformation and imputation of our three datasets before join. Further Discussion of pipeline transformations can be found in Section II below. The workflow of joins is as follows: 
# MAGIC 
# MAGIC ##### Mapping ICAO code to closest weather station
# MAGIC 
# MAGIC `station` data was grouped by `neighbor_code` (`IATA`), reducing the list of stations down to the closest station to each `IATA`, given the provided distance metric. This yielded a two-column mapping table with a unique set of `ICAO` and it's closest weather `station_id`. This new table is referred to as `station_mapping`
# MAGIC 
# MAGIC Flight data was then augmented by joining weather data, using the `stations` table as a mapper between airports/cities in `airlines` and `weather`. Before left-joining `station` on `airlines`, three-letter IATA codes in flight data were mapped to 4-letter ICAO codes. 
# MAGIC 
# MAGIC [GL to include join chunk here]
# MAGIC 
# MAGIC ##### Joining Airline and Station Data (Join 1)
# MAGIC 
# MAGIC `station_mapping` is left joined on `airlines` by `ICAO`. No explicit distance calculations were necessary, as `station_mapping` provided the closest weather station for each airport. 
# MAGIC 
# MAGIC [GL to include join chunk here]
# MAGIC 
# MAGIC ##### Joining Weather data on Airline Data with Station Mappings (Join 2)
# MAGIC 
# MAGIC Once each flight observation was mapped to a it's relevant `station_id`, hourly weather data was left-joined on hourly airline data, keying on `station_id` and `CRS_DEP_TIME_UTC`. During join, all weather observations were lagged by 3 hours to eliminate leakage on hourly-rounded data. Further discussion of transformations required for hourly airline and weather data are found below in Section II.
# MAGIC 
# MAGIC [GL to include join chunk here]

# COMMAND ----------

displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/join.png'>")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Values
# MAGIC 
# MAGIC As discussed in Phase I EDA, features with 70% or higher representation of null values were automatically dropped from our dataset. For remaining features, we followed two processes for treating missing data: 
# MAGIC   1. Removed observatoins considered 'Erroneous' from weather dataset
# MAGIC   2. Imputed missing values using 7-day look back by current date and weather station. 
# MAGIC   
# MAGIC These two techniques allowed for maximum amount of data retention. Discussion of observation imputation methodology is included below in Section II.Imputation. 
# MAGIC 
# MAGIC 
# MAGIC #### Features of Interest
# MAGIC 
# MAGIC The below list of features have been included in our initial models. Following removal of features based on null value representation, features were were further reduced specifically with leakage in mind. The following list of categorical and continuous variables provide us with the most pre-flight information, and are all beleived to be correlated with our variable of interest - `DEP_DEL15`.
# MAGIC 
# MAGIC **Continuous Features:** 
# MAGIC     
# MAGIC     - 'AIRPORT_LAT_ORIGIN',
# MAGIC     - 'AIRPORT_LONG_ORIGIN',
# MAGIC     - 'AIRPORT_LAT_DEST',
# MAGIC     - 'AIRPORT_LONG_DEST',    
# MAGIC     - 'ORIGIN_WEATHER_ELEVATION', 
# MAGIC     - 'ORIGIN_WEATHER_WND_DIRECTION_ANGLE-AVG',
# MAGIC     - 'ORIGIN_WEATHER_WND_SPEED_RATE-AVG',
# MAGIC     - 'ORIGIN_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
# MAGIC     - 'ORIGIN_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
# MAGIC     - ~~'ORIGIN_WEATHER_TMP_AIR_TEMP-AVG',~~ (collinear with  with ORIGIN_WEATHER_DEW_POINT_TEMP-AVG)
# MAGIC     - 'ORIGIN_WEATHER_DEW_POINT_TEMP-AVG',
# MAGIC     - 'ORIGIN_WEATHER_SLP_SEA_LEVEL_PRES-AVG',
# MAGIC     - 'DEST_WEATHER_ELEVATION',
# MAGIC     - 'DEST_WEATHER_WND_DIRECTION_ANGLE-AVG',
# MAGIC     - 'DEST_WEATHER_WND_SPEED_RATE-AVG',
# MAGIC     - 'DEST_WEATHER_CIG_CEILING_HEIGHT_DIMENSION-AVG',
# MAGIC     - 'DEST_WEATHER_VIS_DISTANCE_DIMENSION-AVG',
# MAGIC     - 'DEST_WEATHER_TMP_AIR_TEMP-AVG',
# MAGIC     - 'DEST_WEATHER_DEW_POINT_TEMP-AVG',
# MAGIC     - 'DEST_WEATHER_SLP_SEA_LEVEL_PRES-AVG'
# MAGIC 
# MAGIC **Categorical Features**
# MAGIC     
# MAGIC     - 'YEAR',
# MAGIC     - 'QUARTER',
# MAGIC     - 'MONTH',
# MAGIC     - 'DAY_OF_MONTH',
# MAGIC     - 'DAY_OF_WEEK',
# MAGIC     - 'FL_DATE',
# MAGIC     - 'CRS_DEP_TIME_UTC_HOUR',
# MAGIC     - 'CRS_DEP_TIME_HOUR',    
# MAGIC     - 'OP_UNIQUE_CARRIER',
# MAGIC     - 'ORIGIN_AIRPORT_ID',
# MAGIC     - 'ORIGIN',
# MAGIC     - 'DEST_AIRPORT_ID',
# MAGIC     - 'DEST',
# MAGIC     - 'DEP_TIME_BLK',
# MAGIC     - 'ARR_TIME_BLK',
# MAGIC     - 'CRS_ELAPSED_TIME',
# MAGIC     - 'ID',
# MAGIC 
# MAGIC ##### Comments on potentionally useful features that were excluded
# MAGIC 
# MAGIC 
# MAGIC #### Cross-validation and GridSearch
# MAGIC 
# MAGIC Our current implementation of cross-validation is a sliding window cross-validation with 5 windows. Each fold does not contain any overlap, and is split 80/20 between train and validation. This approach, as pictured below, ensures there is no leakage, or never using future data to inform predictions of past data. 
# MAGIC 
# MAGIC Following implementation of gridsearch, we can leverage our 5-fold CV to search over sets of hyperparameters, and choose the optimal combination that maximizes F0.5. Our gridsearch implementation allows for an array of N hyperparameters, with M different values. Over each fold and grid-search set, the parameters that produce the highest score will be saved. Following gridsearch, the model is trained over a full 2015 - 2018 train set, with the chosen hyperparameters. Finally, testing is applied on the 2019 set. 

# COMMAND ----------

displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/cv_example.png'>")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion of Challenges
# MAGIC 
# MAGIC Throughout the course of the project, some of our largest challenges have to do with the time constraints related to running gridsearch and training models over large chunks of data. Many of our tasks take 6+ hours at this point, some in the 15 hour range. 
# MAGIC 
# MAGIC Additionally, we are facing low F0.5 scores despite numerous feature transformations, hyperparameter combinations, and inclusion of engineered features. 
# MAGIC 
# MAGIC Data leakage strategies have proved challenging, but are important to thoroughly address given our requirement to predict 2 hrs before flights take off. 
# MAGIC 
# MAGIC Finally, we have learned there is a specific order in which transformations must occur. For example, training on a dataset with undersampling applied both to train and val provided inflated val results. Once this issue was addressed, val results returned back to our previous levels.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC [9] https://towardsdatascience.com/csv-files-for-storage-no-thanks-theres-a-better-option-72c78a414d1d
# MAGIC 
# MAGIC # II. Feature Engineering & Transformations
# MAGIC 
# MAGIC #### Treatment of Null Values
# MAGIC 
# MAGIC Our weather data high high volumes of missing observations throughout our variables of interest. Each of these variables had a different encoding for missing data, represented as a series of '9's' of different lengths. Before imputing treating nulls, we standardized these missing codes by replacing the different '9' codes explicitly with 'null'. Additionally, for each weather variable we referenced the codebook for acceptable ranges, and replaced values with 'null' when they fell outside the defined range. Finally, data that was marked as 'erroneous' were also replaced with null. 
# MAGIC 
# MAGIC These three pre-processing steps then allowed us to then study and impute null values. Our methodology included a 7 day look-back window, wherein an average of the previous week replaced a missing value where data was available. This treatment was applied to all continuous features in our dataset. Following imputation, we resolved all but 2.5% of missing value errors across our large 2015 sample. The remaining observations with null values were excluded from our dataset, after confirming missingness was reasonably random across a variety of different feature groupings - like carrier and day of week. 
# MAGIC 
# MAGIC #### Creation/Transformation of Features
# MAGIC 
# MAGIC In order to map airline data to weather data, airline `CRS_DEP_TIME` required conversion to UTC, as weather timestamps were represented in UTC. In addition to this conversion, we also had to lag our `CRS_DEP_TIME_UTC` variable by two hours, to account for the frozen window of two hours before flight departure at inference time. In addition to UTC time conversion for airline observations, the minutes for each `CRS_DEP_TIME_UTC` were dropped, as a simplifying assumption. There is no assurance that for each flight time a weather observation would be also available for that exact timestamp. Similarly, weather observations were also reduced down to an hourly timestamp. Unlike airline data, we were unable to just drop the minutes - because each hour in the weather dataset had multiple observations. To minimize the probability of leakage, we took the first observation for each hour to represent the 
# MAGIC 
# MAGIC Some features within our dataset like `WND`, `CIG`, `VIS`, `TMP`, `DEW`, and `SLP` were composite - and were split into relevant columns rather than retaining the concatenated values. 
# MAGIC 
# MAGIC Features were standardized to remove adverse scale effects in training. [RH to expand further]
# MAGIC 
# MAGIC ##### One-Hot encoding
# MAGIC 
# MAGIC One-hot encoding is employed to deal with non-numerical features. This technique is important as it allows our machine learning model to make best use of categorical information. Current implementation includes one-hot encoding of all categorical variables, which expands our feature list greatly. Future improvements will include reduction of these binary classifications using methodology such as Breimann's approach, and vectorizing one-hot representation to collapse back down to a single feature. 
# MAGIC 
# MAGIC 
# MAGIC ##### Newly Engineered Features
# MAGIC 
# MAGIC In addition to feature transformations, three new feature types were created to increase the power of our model. 
# MAGIC 
# MAGIC 1. Prior leg departure/arrival delay: Binary - 0 when previous flight was on time at arrival/departure, 1 when delayed. Expectation: if prior flights are delayed, the next flight has a higher probability of delay. 
# MAGIC 2. Airport stress @ departure/arrival time: Numeric - measure of business @ airport over last 24 hours. Expectation: airports with higher stress will be correllated with higher frequency of delay. 
# MAGIC 3. Pagerank: Numeric - measure of centrality within the US flight system Grid. Expectation: more central hubs will receive higher amounts of delays, each delay will propagate further throughout the network than an airport with a lower pagerank. 
# MAGIC 
# MAGIC ##### Vectorization
# MAGIC 
# MAGIC Following feature transformation and engineering, the final set of features was ran through a vectorizer to simplify train and reduce computation complexity. The set of features is represented as a sparse vector, rather than a full representatino of the feature set. 

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Algorithm Exploration
# MAGIC 
# MAGIC ### Baseline
# MAGIC Our choice of baseline is Logistic Regression, as it is a fairly simple implementation for binary classification. Our Initial model was reached using undersampling and a regularization parameter of .01, following the above pipeline transformations. For this model our train horizon is January - September 2015, and validation horizon is October - December 2015. 
# MAGIC 
# MAGIC ### Follow-on Models
# MAGIC As we move into tuning and modeling phases, we included quite a few combinations of both models, hyperparameters, and new features. By the end of phase III, we have reached an F0.5 score of .477 training over the full dataset. Summary of models is included below. 
# MAGIC 
# MAGIC ### Results

# COMMAND ----------

displayHTML("<img src ='/files/shared_uploads/gerrit.lensink@berkeley.edu/phase_3_results.png'>")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Discussion
# MAGIC Our baseline model did a poor job of minimizing false positives, which is one of our main objectives. Following this first pass, we will be focusing on oversampling, sliding window cross-validation, and further hyperparameter tuning. 
# MAGIC 
# MAGIC As we iterated through modeling in Phase III, our main objective was to increase precision, thereby increasing F0.5. Now with a higher precision score (.671), we shift our focus to increasing recall during Phase IV in order to improve F0.5. 

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. Algorithm Implementation
# MAGIC 
# MAGIC As we move into phase IV, we plan ot focus on the following tasks to improve our models forecast ability. 
# MAGIC 
# MAGIC 1. Gridsearch across larger list of parameters
# MAGIC 2. Expand gridsearch to other model frameworks, like Random Forest and Gradient Boosted Trees
# MAGIC 3. Test more intelligent models across whole train dataset
# MAGIC 
# MAGIC [TT to expand with Phase IV]

# COMMAND ----------

# MAGIC %md
# MAGIC # V. Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC # VI. Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC Old feature list: 
# MAGIC     - STATION (string):  Cloud GSOD NAA
# MAGIC - DATE (timestamp): Time stamp in UTC
# MAGIC - SOURCE: (short): Need more analysis to understand this column type. 
# MAGIC - LATITUDE (double): Latitude coordinates
# MAGIC - LONGITUDE (double): Longitude coordinates
# MAGIC - ELEVATION (double): The elevation of a geophysical-point-observation relative to Mean Sea Level (MSL).
# MAGIC - NAME (string): City name
# MAGIC - REPORT_TYPE (string): The code that denotes the type of geophysical surface observation.
# MAGIC - CALL_SIGN (string): Needs more analysis to understand this column type. Most values are 99999.
# MAGIC - QUALITY_CONTROL (string): The name of the quality control process applied to a weather observation.
# MAGIC     - V01 = No A or M Quality Control applied 
# MAGIC     - V02 = Automated Quality Control
# MAGIC     - V03 = subjected to Quality Control
# MAGIC - WND (string): Wind direction and speed
# MAGIC - CIG (string): Ceiling & Visibility
# MAGIC - VIS (string): Visibility observation in meters
# MAGIC - TMP (string): Air temperature in celcius
# MAGIC - DEW (string): The temperature to which a given parcel of air must be cooled at constant pressure and water vapor content in order for saturation to occur.
# MAGIC - SLP (string): The air pressure relative to Mean Sea Level (MSL). Units: Hectopascals
# MAGIC - ~~GA1 (string): The identifier that represents a sky-cover-layer.~~
# MAGIC - ~~GE1 (string): An indicator that denotes the start of a sky-condition-observation data group.~~
# MAGIC - ~~GF1 (string): An indicator that denotes the start of a sky-condition-observation data group.~~
# MAGIC - ~~MA1 (string): The identifier that denotes the start of an atmospheric-pressure-observation data section.~~
# MAGIC - ~~REM (string): Remarks. This column needs more analysis.~~
# MAGIC - ~~GD1 (string): The code that denotes the fraction of the total celestial dome covered by a sky-cover-layer.~~

# COMMAND ----------

