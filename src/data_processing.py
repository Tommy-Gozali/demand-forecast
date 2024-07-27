import numpy as np
import pandas as pd
import os
import itertools

from copy import deepcopy
from utils import DataCleaningService, FeatureEngineering
from create_experiment import CreateExperiment
from get_exog_data import OpenMeteoApi
from sklearn.preprocessing import PolynomialFeatures

cwd = os.path.abspath("")
raw_data_file_path = os.path.join(cwd, './data/power_load_BE_elia_15M_2015_2024.csv')
df = pd.read_csv(raw_data_file_path, delimiter = ";", on_bad_lines="skip")

clean_start_date = "2021-11-01"
clean_end_date   = "2023-06-23"
hour_in_week     = 24*7
hour_in_month    = 24*30

# CLEAN INPUT DATA
cleaning_service = DataCleaningService()
df_clean = cleaning_service.normalize_to_timeseries_df(df = df,
                                          start_date = clean_start_date,
                                          end_date   = clean_end_date,
                                          date_index = "Datetime",
                                          non_float_int_cols=["Datetime", "Resolution code"])

float_columns = [['Total Load', 'Most recent forecast',
                  'Most recent P10', 'Most recent P90', 'Day-ahead 6PM forecast',
                  'Day-ahead 6PM P10', 'Day-ahead 6PM P90', 'Week-ahead forecast']]

df_clean = df_clean.apply(lambda x : x.astype(float) if x.name in float_columns else x)

cleaning_service.check_missing_date(df = df_clean)
# CREATE EXPERIMENT
experiment = CreateTimeSeriesExperiment(data = df_clean,
                              frequency_magnitude_input = 15,
                              frequency_unit_input = "m", #m = minutes, H = hour
                              frequency_magnitude_output = 1,
                              frequency_unit_output = "h", #m = minutes, H = hour
                              forecast_type = "day_ahead",
                              train_size = 0.7)

duration_in_day, duration_in_week, duration_in_month = experiment.basic_durations
train_indexes, val_indexes, test_indexes = experiment.create_train_test_validation_split_round()
steps = experiment.steps
df_clean = experiment.get_input_data()
#experiment.visualize_train_test_validation_split(y = "Total Load")
print(f"Train dates : {train_indexes.min()} --- {train_indexes.max()}  (n={len(train_indexes)})")
print(f"Test dates  : {val_indexes.min()} --- {val_indexes.max()}  (n={len(val_indexes)})")
print(f"Val dates   : {test_indexes.min()} --- {test_indexes.max()}  (n={len(test_indexes)})")

# GET EXOGENOUS DATA
data = deepcopy(df_clean)
brussel_longitude = 50.85
brussel_latitude  = 4.34
brussel_weather_data = OpenMeteoApi(brussel_longitude,
                                    brussel_latitude)
def get_start_end_date(date_list: list):
    return str(date_list[0])[:10], str(date_list[-1])[:10]

start_date_train, end_date_train = get_start_end_date(train_indexes)
start_date_val,  end_date_val    = get_start_end_date(val_indexes)
start_date_test,  end_date_test  = get_start_end_date(test_indexes)

exog_wea_features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "is_day"]

parameters_train = brussel_weather_data.initiate_parameters(start_date = start_date_train,
                                                            end_date = end_date_train,
                                                            parameters = exog_wea_features)
parameters_test = brussel_weather_data.initiate_parameters(start_date = start_date_test,
                                                            end_date = end_date_test,
                                                            parameters = exog_wea_features)
parameters_val = brussel_weather_data.initiate_parameters(start_date = start_date_val,
                                                            end_date = end_date_val,
                                                            parameters = exog_wea_features)

data[exog_wea_features] = brussel_weather_data.get_historical_actual_data(parameters_train)\
                            .combine_first(brussel_weather_data.get_historical_forecast_data(parameters_val))\
                            .combine_first(brussel_weather_data.get_historical_forecast_data(parameters_test))

# FEATURE ENGINEERING
fe = FeatureEngineering()
data = data.rename(columns = {"Total Load":"y"})
basic_temporal_features = ["month", "dayofweek", "hour"]
data[basic_temporal_features] = fe.get_basic_temporal_features(df=data)

day = 1
day_in_week = 7
duration_list   = [day, day_in_week]
agg_method_list = ['mean', 'max', 'min']

duration_agg_method_combinations = list(itertools.product(duration_list, agg_method_list))
rolling_feature_df_list = []
for element in duration_agg_method_combinations:
    data_exog_rolled = fe.get_rolling_data(df = data,
                                            parameter = "temperature_2m",
                                            duration  = element[0],
                                            duration_unit = "D",
                                            agg_method = element[1])
    rolling_feature_df_list.append(data_exog_rolled)

data_exog_rolled = pd.concat(rolling_feature_df_list, axis = 1)
data = pd.concat([data,data_exog_rolled], axis = 1)
rolled_features = data_exog_rolled.columns.tolist()

transformer_poly = PolynomialFeatures(
                       degree           = 2,
                       interaction_only = True,
                       include_bias     = False
                   ).set_output(transform="pandas")

poly_cols = ['temperature_2m','relative_humidity_2m', 'dew_point_2m',
             'apparent_temperature', 'temperature_2m_rolling_mean_1_D', 'temperature_2m_rolling_mean_7_D']
poly_features = transformer_poly.fit_transform(data[poly_cols].dropna())
poly_features = poly_features.drop(columns=poly_cols)
poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
poly_features.columns = poly_features.columns.str.replace(" ", "__")


data_train = data.loc[train_indexes].dropna() #TO DELETE MISSING ROWS DUE TO ROLLED PARAMETERS
data_val   = data.loc[val_indexes]
data_test  = data.loc[test_indexes]

data = pd.concat([data_train, data_val, data_test])

for dataset in [data_train, data_val, data_test]:
    if dataset.isna().sum().sum() > 0:
        print("Missing data!!")

print("New datasets after considering rolled features!")
print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train.index)})")
print(f"Test dates  : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val.index)})")
print(f"Val dates   : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test.index)})")

exog_vars = basic_temporal_features + exog_wea_features + rolled_features

