import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from utils import DataCleaningService, FeatureEngineering
from create_experiment import CreateExperiment
from get_exog_data import OpenMeteoApi

cwd = os.path.abspath("")
raw_data_file_path = os.path.join(cwd, './data/power_load_BE_elia_15M_2015_2024.csv')
df = pd.read_csv(raw_data_file_path, delimiter = ";", on_bad_lines="skip")

clean_start_date = "2021-11-01"
clean_end_date   = "2023-06-23"
hour_in_week     = 24*7

# CLEAN INPUT DATA
cleaning_service = DataCleaningService()
df_clean = cleaning_service.normalize_data(df = df,
                                          start_date = clean_start_date,
                                          end_date   = clean_end_date)
                                          #offset_rolling_feature_in_h = hour_in_week )
cleaning_service.check_missing_date(df = df_clean)
# CREATE EXPERIMENT
experiment = CreateExperiment(data = df_clean,
                              frequency_magnitude_input = 15,
                              frequency_unit_input = "m", #m = minutes, H = hour
                              frequency_magnitude_output = 1,
                              frequency_unit_output = "H", #m = minutes, H = hour
                              forecast_type = "day_ahead",
                              train_size = 0.7)

duration_in_day, duration_in_week, duration_in_month = experiment.basic_durations
train_indexes, val_indexes, test_indexes = experiment.create_train_test_validation_split_round()
steps = experiment.steps
df_clean = experiment.get_input_data()
experiment.visualize_train_test_validation_split(y = "Total Load")
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

exog_wea_parameters = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "is_day"]

parameters_train = brussel_weather_data.initiate_parameters(start_date = start_date_train,
                                                            end_date = end_date_train,
                                                            parameters = exog_wea_parameters)
parameters_test = brussel_weather_data.initiate_parameters(start_date = start_date_test,
                                                            end_date = end_date_test,
                                                            parameters = exog_wea_parameters)
parameters_val = brussel_weather_data.initiate_parameters(start_date = start_date_val,
                                                            end_date = end_date_val,
                                                            parameters = exog_wea_parameters)

data[exog_wea_parameters] = brussel_weather_data.get_historical_actual_data(parameters_train)\
                            .combine_first(brussel_weather_data.get_historical_forecast_data(parameters_val))\
                            .combine_first(brussel_weather_data.get_historical_forecast_data(parameters_test))