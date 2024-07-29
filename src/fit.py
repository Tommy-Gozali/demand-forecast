from xgboost import XGBRegressor # type: ignore
import joblib
import pandas as pd
import os 
from data_processing import (data_train, data_val, exog_vars, steps,
                             duration_in_day, duration_in_week, duration_in_month)
    #data, data_train, data_val, data_val, exog_vars, hour_in_month, steps
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Create and train forecaster
# ==============================================================================
data_train_val = pd.concat([data_train, data_val])

regressor = XGBRegressor(
                            n_estimators=2000,
                            tree_method='gpu_hist',
                            gpu_id=0
                            )
forecaster = ForecasterAutoreg( regressor
               ,
                lags = duration_in_month
             )

forecaster.fit(y=data_train_val['y'], exog=data_train_val[exog_vars])

cwd = os.getcwd()
regressor_name_to_save = str(regressor).split('(')[0]
path = os.path.join(cwd, f'results_gpu_train_{regressor_name_to_save}_{steps}_steps.joblib')
joblib.dump(forecaster, path)