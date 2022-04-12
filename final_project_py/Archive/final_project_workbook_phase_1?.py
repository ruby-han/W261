# Databricks notebook source
# MAGIC %md
# MAGIC # Phases Summaries
# MAGIC > *A brief summary of your progress by addressing the requirements for each phase. These should be concise, one or two paragraphs. There is a character limit in the form of 1000 for these*
# MAGIC 
# MAGIC ### Phase 1 (Due 3/13 @ Midnight)
# MAGIC > Our initial EDA tasks were focused on understanding the data and features, exploring null values, and determining metrics for evaluation of our future model. Early in the process we were able to identify that our data is heavily imbalanced - with only 20% of our data classified as a flight delay. Second, we narrowed down the list of features based on proportion of null values. We dropped roughly half of the features in our `airlines` dataset, and 90% of the features in `weather`.
# MAGIC 
# MAGIC > We also focused on how we would augment our flight data to increase forecast performance. To do this, include two different auxiliary tables - `stations` which is provided, and an airport to lat/long mapping, pulled from the web. A series of three joins described below will be necessary to create our full dataset. 
# MAGIC 
# MAGIC > Finally, we decided that the metric best suited to help airlines reduce costs would be precision. This metric directly measures the share of false positives in relation to total positives, and gives us a mechanism to measure how often we incorrectly predict a delay when a delay does not actually occur. We believe these false positives to be the most costly mistakes. (*~972 chars not including spaces*)
# MAGIC 
# MAGIC ### Phase 2 (Due 3/20 @ midnight)
# MAGIC 
# MAGIC ### Phase 3 (Due 4/3 @ midnight)

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Formulation
# MAGIC > *You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etc.. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.*
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
# MAGIC # Phase 1 - Describe datasets, joins, task, and metrics
# MAGIC 
# MAGIC *Relocated contents of this section to "EDA & Discussion of Challenges"*

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA & Discussion of Challenges
# MAGIC ### Phase 1 - Describe datasets, joins, task, and metrics
# MAGIC 
# MAGIC External Notebook links:
# MAGIC * [Initial EDA on Airline, weather, and stations datasets](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885897629/command/1038316885897636)
# MAGIC * [Further EDA on weather dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885900748/command/1038316885900753)
# MAGIC 
# MAGIC 
# MAGIC #### 1. Explain data (i.e.., simple exploratory analysis of various fields, such as the semantic as well as intrinsic meaning of ranges, null values, categorical/numerical, mean/std.dev to normalize and/or scale inputs). Identify any missing or corrupt (i.e., outlier) data.
# MAGIC **Airline Data**
# MAGIC > This dataset consists of 109 different columns describing airline data, with two different subsets - Q1 of 2015, or Q1 + Q2 of 2015. For our initial EDA we focused on the 6 month dataset, assuming it is fairly representative of the overall dataset. This 6-month subset had 337k different observations. Our target variable is `DEP_DEL15`, which is a binary variable that is '1' when the given flight is delayed by 15 mins or more, and '0' when there is no delay or the delay is less than 15 minutes. Across the entire sample, we've observed an average delay of 14 minutes, with 36 minutes standard deviation. Our group will pay special attention to the fact that the dataset is highly imbalanced - with only ~21% of the flight records including a delay of 15 or more minutes. 
# MAGIC 
# MAGIC > During data integrity checks, ~48 columns were identified as consisting of 90% or greater null values. These columns were dropped as they contained very little userful information and treatment of missing values would be innacurate. Additionally, we are able to drop columns related to arrival information, which further allowed us to reduce the size of our dataset. These columns contained data that would be considered leakage, i.e. at departure time - 2 hrs, we would not know about different arrival metrics for the same leg of the flight - therefore they must be removed. 
# MAGIC 
# MAGIC > In addition to checks for null values - we also focused on identification of duplicate data. This was accomplished by creating a new unique_id column, which is a concatenation of `FL_DATE`, `TAIL_NUM`, `ORIGIN`, and `CRS_DEP_TIME`. Duplicates identified are commonly related to cancelled flights that are later replaced by flights to a different destination, with same `FL_DATE`, `TAIL_NUM`, and `ORIGIN`. Cancelled flights were dropped, and replacement flights were retained. 
# MAGIC 
# MAGIC **Weather Data**: 
# MAGIC 
# MAGIC > The weather dataset (6 months) consists of 29,823,926 (29.8 million) rows and 177 columns. Across the 177 columns, 156 columns consist of 60% or greater null values. This high proportion of nulls motivates removal of these columns - as treatment of this many nulls would be inaccurate. Additional decoding was also included, with features like `WND`, `CIG`, `VIS`, `TMP`, `DEW`, and `SLP` were composite - and were split into relevant columns rather than retaining the concatenated values. 
# MAGIC 
# MAGIC During initial EDA, any columns with more than 70% null values/blank strings/`nan` were dropped. The columns of interest which will be used for analysis are below:
# MAGIC 
# MAGIC - STATION (string):  Cloud GSOD NAA
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
# MAGIC - GA1 (string): The identifier that represents a sky-cover-layer.
# MAGIC - GE1 (string): An indicator that denotes the start of a sky-condition-observation data group.
# MAGIC - GF1 (string): An indicator that denotes the start of a sky-condition-observation data group.
# MAGIC - MA1 (string): The identifier that denotes the start of an atmospheric-pressure-observation data section.
# MAGIC - REM (string): Remarks. This column needs more analysis.
# MAGIC - GD1 (string): The code that denotes the fraction of the total celestial dome covered by a sky-cover-layer.
# MAGIC 
# MAGIC **Stations Data**: 
# MAGIC 
# MAGIC > The stations dataset will be used to connect the closest weather station to each airport in the dataset. This dataset consists of 5 million unique weather stations classified by latitude and longitude. Each observation has non-null values across the 12 features, so no treatment of null values is required. Latitude and longitude will be used to map the top ‘closest’ weather stations to each airport. Additional mappings will be necessary for the airport dataset, as it’s current state does not include latitude or longitude information. 
# MAGIC 
# MAGIC #### 2. Define the outcome (i.e., the evaluation metric and the target) precisely, including mathematical formulas.
# MAGIC 
# MAGIC > Our model will predict whether a flight is expected to be delayed by more than 15 minutes up to two hours before the scheduled departure time. Specifically, our target variable is `DEP_DEL15`, where '1' represents a flight delay of 15 minutes or greater, and 0 represents an on-time departure (or an 'insignificant' delay of less than 15 minutes). 
# MAGIC 
# MAGIC > Because our model is built to reduce costs specifically for the airline, we focus on minimzing the rate at which we predict false positives, which is defined as a prediction of a delay, when the flight will not actually require a delay. We have chosen to focus on this specifically because as a false positive would trigger many costs for the airline like lost revenue due to empty seats, flight changes by customers, etc. We define precision as \\( \frac {FP}{FP + TP} \\), where \\(FP\\) = False Positive and \\(TP \\) = True Positive
# MAGIC 
# MAGIC #### 3. How do you ingest the CSV files and represent them efficiently? (think about different file formats)
# MAGIC 
# MAGIC > Where possible, files will be loaded and stored as parquet files rather than csv formats. In general, parquet's column-wise format is more efficient both from a storage perspective as well as read speed. [1] Data will be stored/processed across RDDs and pyspark dataframes rather than Pandas - which allows for distributed computing. 
# MAGIC 
# MAGIC #### 4. Join relevant datasets
# MAGIC     
# MAGIC **a. Describe what tables to join**
# MAGIC 
# MAGIC > We will be augmenting our flight data by joining weather data, using the `stations` table as a mapper between airports/cities in `airlines` and `weather`. Additional mapping tables will be required to map airlines to stations - as there is no lat/long data included in the airports dataset, we will call this `airport`. 
# MAGIC      
# MAGIC **b. Describe the workflow for achieving the joins (what are the keys, type of join)**
# MAGIC     
# MAGIC > All data will be left joined on `weather`, as we do not wish to ignore observations in the `airlines` table if weather data is not available. Instead, we will treat these missing values as discussed in C below. The flow of joins would be as follows: 
# MAGIC     
# MAGIC > * Left join `airport` on `airlines`, keying on both `origin` and `destination`. added columns will be `org_lat`, `org_long`, `dest_lat`, and `dest_long` from the `airport` table. 
# MAGIC     
# MAGIC > * Left join `stations` on `airlines`, based on calculation of closest 3 stations by lat/long. This will result in three new features each for `origin` and `destination` - `org_station1`, `org_station2`, and `org_station3`. Once each of these stations are mapped - we will be able to pull in relevant columns from the `weather` table. Weather information will require a more complex key related to both the org_station, as well as date and time - pulling the weather data for departure time - 2 hours. 
# MAGIC     
# MAGIC **c. Steps to deal with potential missing values**
# MAGIC 
# MAGIC > Because we want to retain all of our airline data even if weather data is missing, we will fill missing weather values rather than ignore. For example if temperature data is missing on a given day, we will replace it with the monthly average for the given month.
# MAGIC     
# MAGIC #### 5. Checkpoint the data after joining to avoid wasting time and resources!
# MAGIC 
# MAGIC #### 6. Split the data train/validation/test - making sure that no leaks occur, for example: normalize using the training statistics. 
# MAGIC *HINT: see Additional resources 2 below: Cross-validation in Time Series (very different to regular cross-validation*
# MAGIC   
# MAGIC > Our initial Train/Val/Test split buckets years 2015, 2016, and 2017 into train, 2018 into validation, and 2019 into test. Due to the time series nature of the data, we cannot randomly sample data into natural 80/20 train test splits, or even run classic k-folds. In order to run more robuts models in future phases while protecting against data leakage - the team will explore time-series-specific versoins of cross-validation, such as slidingWindowSplitter. This technique allows for cross-validation that is restricted to training on current data, and testing on future data which protects against data leakage. 
# MAGIC   
# MAGIC [1] https://towardsdatascience.com/csv-files-for-storage-no-thanks-theres-a-better-option-72c78a414d1d

# COMMAND ----------

# MAGIC %md 
# MAGIC # Feature Engineering
# MAGIC 
# MAGIC Phase 2: talk about weather data re-mapping/splits here

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorighm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC # Application of Course Concepts

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Aditya is typing here.
# MAGIC 
# MAGIC 
# MAGIC - STATION (string):  Cloud GSOD NAA
# MAGIC - DATE (timestamp): Time stamp in UTC
# MAGIC - SOURCE: (short): **NOT SURE** most values look like 4
# MAGIC - LATITUDE (double): Latitude coordinates
# MAGIC - LONGITUDE (double): Longitude coordinates
# MAGIC - ELEVATION (double): The elevation of a GEOPHYSICAL-POINT-OBSERVATION relative to Mean Sea Level (MSL).
# MAGIC - NAME (string): City Name
# MAGIC - REPORT_TYPE (string): The code that denotes the type of geophysical surface observation.
# MAGIC - CALL_SIGN (string): **NOT SURE** Most values are 99999.
# MAGIC - QUALITY_CONTROL (string): The name of the quality control process applied to a weather observation.
# MAGIC     - V01 = No A or M Quality Control applied 
# MAGIC     - V02 = Automated Quality Control
# MAGIC     - V03 = subjected to Quality Control
# MAGIC - WND (string): Wind direction and speed
# MAGIC - CIG (string): **NOT SURE** Ceiling & Visibility ?
# MAGIC - VIS (string): Visibility observation in meters
# MAGIC - TMP (string): Air temperature in celcius
# MAGIC - DEW (string): The temperature to which a given parcel of air must be cooled at constant pressure and water vapor content in order for saturation to occur.
# MAGIC - SLP (string): The air pressure relative to Mean Sea Level (MSL). UNITS: Hectopascals
# MAGIC - GA1 (string): The identifier that represents a SKY-COVER-LAYER.
# MAGIC - GE1 (string): An indicator that denotes the start of a SKY-CONDITION-OBSERVATION data group.
# MAGIC - GF1 (string): An indicator that denotes the start of a SKY-CONDITION-OBSERVATION data group.
# MAGIC - MA1 (string): The identifier that denotes the start of an ATMOSPHERIC-PRESSURE-OBSERVATION data section.
# MAGIC - REM (string): Remarks **Not sure** 
# MAGIC - GD1 (string): The code that denotes the fraction of the total celestial dome covered by a SKY-COVER-LAYER.

# COMMAND ----------

