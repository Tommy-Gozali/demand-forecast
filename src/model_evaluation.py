import os
import joblib
import pandas as pd
from data_processing import data, data_train, data_val, data_test, exog_vars, hour_in_week
from sklearn.metrics import mean_absolute_percentage_error
from forecast import BacktestingForecaster

cwd = os.path.dirname(__file__)
modelling_file_path = os.path.join(cwd, './modelling/model/forecaster.joblib')
backtest_result_file_path = os.path.join(cwd, '../data/backtest_result.csv')

forecaster = joblib.load(modelling_file_path)
regressor_name = str(forecaster.regressor).split('(')[0]

data_train_val = pd.concat([data_train, data_val])

bf = BacktestingForecaster(
    forecaster = forecaster,
    y = data["y"],
    steps = hour_in_week,
    exog = data[exog_vars],
    metric = mean_absolute_percentage_error,
    initial_train_size = len(data_train),
    refit = True,
)

def backtest():
    metric, predictions = bf.backtesting_forecaster()
    print(f"Test error (MAPE) for {regressor_name}: {metric*100}%")
    predictions.index = data_test.index
    data_test[regressor_name] = predictions
    """
    for i, fold in enumerate(predictions):
        print(f"Fold {i}: {fold}")
    """    
    return data_test.to_csv(backtest_result_file_path, index = False)

#backtest()
folds_int = bf._create_backtesting_folds()
folds = bf.get_backtesting_folds(
    backtest_indexes= folds_int,
    backtest_result_df= pd.read_csv(backtest_result_file_path),
    regressor_name= regressor_name,)

