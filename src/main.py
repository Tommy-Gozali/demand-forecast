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

from data_processing import data, data_train, data_test, data_val, exog_vars, hour_in_month

# Create and train forecaster
# ==============================================================================

regressor  = RandomForestRegressor(max_depth=10, n_estimators=50)
forecaster = ForecasterAutoreg(
                 regressor = regressor,
                 lags      = hour_in_month
             )

forecaster.fit(y=data_train['y'], exog=data_train[exog_vars])

#Save the trained model
cwd = os.path.dirname(__file__)
modelling_file_path = os.path.join(cwd, './modelling/model/forecaster.joblib')
joblib.dump(forecaster, modelling_file_path)
