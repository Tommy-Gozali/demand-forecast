import os
import joblib
from data_processing import steps, data, data_train, data_test, data_val, exog_vars, hour_in_week
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.model_selection import backtesting_forecaster
import pandas as pd

cwd = os.path.dirname(__file__)
modelling_file_path = os.path.join(cwd, "results_gpu_train_XGBRegressor_24_steps.joblib")
backtest_result_file_path = os.path.join(cwd, '../data/backtest_result.csv')

forecaster = joblib.load(modelling_file_path)
regressor_name = str(forecaster.regressor).split('(')[0]
data_train_val = pd.concat([data_train, data_val])

# Predictions
# ==============================================================================
def predict_single_forecast():
    predictions = forecaster.predict(steps = steps,
                                    exog  = data_test[exog_vars])

    predictions.index = data_test.index
    print(forecaster.__dict__)

    data_test[regressor_name] = predictions
    fig = data_test[["y", "Most recent forecast", regressor_name, "Week-ahead forecast"]].plot.line()
    fig.show()

    for col in ["Most recent forecast",regressor_name , "Week-ahead forecast"]:
        error_mape = mean_absolute_percentage_error(
                        y_true = data_test['y'],
                        y_pred = data_test[col]
                    )
    return print(f"Test error (MAPE) for {col}: {error_mape*100}%")

def backtest():
    metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data['y'],
                          steps              = hour_in_week,
                          exog               = data[exog_vars],
                          metric             = 'mean_absolute_percentage_error',
                          initial_train_size = len(data_train_val),
                          refit              = True,
                          n_jobs             = 'auto',
                          verbose            = True,
                          show_progress      = True
                      )
    #print(f"Test error (MAPE) for {col}: {metric*100}%")
    predictions.index = data_test.index
    data_test[regressor_name] = predictions

    for i, fold in enumerate(predictions):
        print(f"Fold {i}: {fold}")
        
    return data_test.to_csv(backtest_result_file_path, index = False)

backtest()