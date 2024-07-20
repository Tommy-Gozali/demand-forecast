import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from get_exog_data import *
from utils import *
from sklearn.metrics import mean_absolute_percentage_error


# Warnings configuration
# ==============================================================================
#import warnings
#warnings.filterwarnings('ignore')
pd.options.plotting.backend = "plotly"

import os
import pandas as pd
cwd = os.path.dirname(__file__)
# Construct the full path to the CSV file
file_path = os.path.join(cwd, '../data/power_load_BE_elia_15M_2015_2024.csv')

df = pd.read_csv(file_path, delimiter = ";", on_bad_lines="skip")

df = df[df["Datetime"].str.contains("2023|2022")]
clean_start_date = "2021-11-01"
clean_end_date   = "2023-06-23"
def normalize_data(df: pd.DataFrame,
                   start_date: str,
                   end_date  : str,
                   offset_rolling_feature_in_h: int):

    real_start_date = str(pd.to_datetime(start_date, format = "%Y-%m-%d") - pd.Timedelta(offset_rolling_feature_in_h, "h"))[:10]
    real_end_date   = end_date
    date_range = f"{real_start_date}:{real_end_date}"
    df.index = pd.to_datetime(df["Datetime"], utc = True)
    df = df.drop(columns = ["Datetime", "Resolution code"]).sort_index()
    df = df.loc[real_start_date:real_end_date]
    return df

clean_data = ["2021-11-01:2023-06-23","2015-01-01:2021-10-31 01:45:00"]
hour_in_week = 24*7
start_date   = "2022-06-01"
end_date     = "2023-06-15"
df_2023 = normalize_data(df = df, start_date= start_date, end_date=end_date,
               offset_rolling_feature_in_h = hour_in_week )

missing_date = pd.date_range(start = df_2023.index[0],
                             end   = df_2023.index[-1] ).difference(df_2023.index)
missing_date

def find_bad_float_data(series):
    bad_rows = []
    for index, row in enumerate(series):
        try:
            float(row)
        except:
            bad_rows.append({index:row})

    return bad_rows

df_2023 = df_2023.apply(lambda x : x.astype(float))

def return_missing_index(series):
    missing_index = np.where(series.isna())
    return [np.amin(missing_index), np.amax(missing_index)]

missing_list = []
for col in df_2023.columns:
    try:
        missing_list.append({col:return_missing_index(df_2023[col])})
    except:
        print(col)
df_2023_interp = df_2023.interpolate()
data = df_2023_interp #.loc["2023-01-01 00:00:00":"2023-06-01 00:00:00"]
data = data.resample("h").mean()
data = data.rename(columns = {"Total Load":"y"})

data["month"] = data.index.month
data["dayofweek"] = data.index.dayofweek
data["hour"] = data.index.hour
hour_in_day   = 24
hour_in_week  = 24*7
hour_in_month = hour_in_week * 4
folds = 8
steps = hour_in_week * 8

data_train_index = data.index[:-steps]
data_test_index  = data.index[-steps:]

print(f"Train dates : {data_train_index.min()} --- {data_train_index.max()}  (n={len(data_train_index)})")
print(f"Test dates  : {data_test_index.min()} --- {data_test_index.max()}  (n={len(data_test_index)})")

brussel_longitude = 50.85
brussel_latitude  = 4.34
brussel_weather_data = OpenMeteoApi(brussel_longitude,
                                    brussel_latitude)
def get_start_end_date(date_list: list):
  return str(date_list[0])[:10], str(date_list[-1])[:10]

start_date_train, end_date_train = get_start_end_date(data_train_index)
start_date_test,  end_date_test  = get_start_end_date(data_test_index)
parameters = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "is_day"]#, "weather_code"]

parameters_train = brussel_weather_data.initiate_parameters(start_date = start_date_train,
                                                            end_date = end_date_train,
                                                            parameters = parameters)
parameters_test = brussel_weather_data.initiate_parameters(start_date = start_date_test,
                                                            end_date = end_date_test,
                                                            parameters = parameters)
data[parameters] = brussel_weather_data.get_historical_forecast_data(parameters_train)\
    .combine_first(brussel_weather_data.get_historical_forecast_data(parameters_test))
"""
data_exog_wea_train, data_exog_wea_test = brussel_weather_data.get_historical_forecast_data(parameters_train),\
                                          brussel_weather_data.get_historical_forecast_data(parameters_test)
                                          """

import itertools

day = 1
day_in_week = 7
duration_list   = [day, day_in_week]
agg_method_list = ['mean', 'max', 'min']

duration_agg_method_combinations = list(itertools.product(duration_list, agg_method_list))

fe = FeatureEngineering()
rolling_feature_df_list = []
for element in duration_agg_method_combinations:
  data_exog_rolled = fe.get_rolling_data(df = data,
                                    parameter = "temperature_2m",
                                    duration = element[0],
                                    duration_unit = "D",
                                    agg_method = element[1])
  rolling_feature_df_list.append(data_exog_rolled)

data_exog_rolled = pd.concat(rolling_feature_df_list, axis = 1)
data = pd.concat([data,data_exog_rolled], axis = 1)
rolled_parameters = data_exog_rolled.columns.tolist()

data_train = data.loc[data_train_index].loc[start_date:end_date]
data_test  = data.loc[data_test_index].loc[start_date:end_date]
data = data.loc[start_date:end_date]

print(data_train.isna().sum() + data_test.isna().sum())

exog_vars = ["month", "dayofweek", "hour"] + parameters + rolled_parameters

# Create and train forecaster
# ==============================================================================

regressor  = RandomForestRegressor(max_depth=10, n_estimators=50)
regressor_name = str(regressor).split('()')[0]
forecaster = ForecasterAutoreg(
                 regressor = regressor,
                 lags      = hour_in_month
             )

forecaster.fit(y=data_train['y'], exog=data_train[exog_vars])

# Predictions
# ==============================================================================
predictions = forecaster.predict(steps = steps,
                                 exog  = data_test[exog_vars])
predictions.index = data_test.index
data_test[regressor_name] = predictions
fig = data_test[["y", "Most recent forecast", regressor_name, "Week-ahead forecast"]].plot.line()
fig.show()

for col in ["Most recent forecast",regressor_name , "Week-ahead forecast"]:
    error_mape = mean_absolute_percentage_error(
                    y_true = data_test['y'],
                    y_pred = data_test[col]
                )

print(f"Test error (MAPE) for {col}: {error_mape*100}%")