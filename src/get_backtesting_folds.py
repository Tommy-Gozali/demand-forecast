import os
import pandas as pd
import joblib
from data_processing import (data, data_train, data_val, exog_vars, steps,
                             duration_in_day, duration_in_week, duration_in_month, hour_in_week)

cwd = os.getcwd()
backtest_result_file_path = os.path.join(cwd, './data/backtest_result.csv')
modelling_file_path = os.path.join(cwd, "./src/results_gpu_train_XGBRegressor_24_steps.joblib") #'./modelling/model/forecaster.joblib')

forecaster = joblib.load(modelling_file_path)
df = pd.read_csv(backtest_result_file_path)


import forecast
from sklearn.metrics import mean_absolute_percentage_error

data_train_val = pd.concat([data_train, data_val])

bf = forecast.BacktestingForecaster(
    forecaster = forecaster,
    y = data["y"],
    steps = steps,
    initial_train_size = len(data_train),
    metric = mean_absolute_percentage_error,
    refit = True,
)