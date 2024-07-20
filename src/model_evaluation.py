import os
import joblib
from data_processing import steps, data_test, exog_vars
from sklearn.metrics import mean_absolute_percentage_error

cwd = os.path.dirname(__file__)
modelling_file_path = os.path.join(cwd, './modelling/model/forecaster.joblib')

forecaster = joblib.load(modelling_file_path)
regressor_name = str(forecaster.regressor).split('()')[0]


# Predictions
# ==============================================================================
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

print(f"Test error (MAPE) for {col}: {error_mape*100}%")
