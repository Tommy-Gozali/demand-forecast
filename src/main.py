import os

# Modeling and Forecasting
# ==============================================================================
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor 


from data_processing import (data_test, exog_vars, steps
    #data, data_train, data_val, data_val, exog_vars, hour_in_month, steps
)
import plotly.express as px


"""
# Create and train forecaster
# ==============================================================================

regressor  = RandomForestRegressor(max_depth=10, n_estimators=50)
forecaster = ForecasterAutoreg(
                 regressor = regressor,
                 lags      = hour_in_month
             )

forecaster.fit(y=data_train['y'], exog=data_train[exog_vars])

joblib.dump(forecaster, modelling_file_path)

"""

# Predictions
# ==============================================================================
cwd = os.getcwd()
modelling_file_path = os.path.join(cwd, "src/modelling/model/results_gpu_train_XGBRegressor_24_steps.joblib") #'./modelling/model/forecaster.joblib')

forecaster = joblib.load(modelling_file_path)
regressor_name = str(forecaster.regressor).split('(')[0]

def predict_single_forecast(data_val: pd.DataFrame, steps: int, exog_vars: list):
    predictions = forecaster.predict(steps = steps,
                                    exog  = data_val[exog_vars])

    predictions.index = data_val.iloc[:steps].index

    data_val[regressor_name] = predictions
    data_val = (data_val.dropna(subset= [regressor_name]).reset_index())
    data_val_melted = pd.melt(data_val, 
                              id_vars = ["Datetime"],
                              value_vars=["y", "Most recent forecast", regressor_name, "Week-ahead forecast"],
                              var_name = "models",
                              value_name = "total load [MW]")

    
    fig = px.line(data_val_melted, y="total load [MW]", x= "Datetime", color="models")

    for col in ["Most recent forecast",regressor_name , "Week-ahead forecast"]:
        error_mape = mean_absolute_percentage_error(
                        y_true = data_val['y'],
                        y_pred = data_val[col]
                    )
        print(f"Test error (MAPE) for {col}: {error_mape*100}%")
    fig.show()
    return fig

predict_single_forecast(
    data_val = data_test, 
    steps = steps, 
    exog_vars = exog_vars)