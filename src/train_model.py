import pandas as pd
import matplotlib.pyplot as plt
import torch # type: ignore
import os
import psutil
from xgboost import XGBRegressor # type: ignore


from data_processing import (data_train, data_val, exog_vars, steps,
                             duration_in_day, duration_in_week, duration_in_month)
    #data, data_train, data_val, data_val, exog_vars, hour_in_month, steps
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg

data_train_val = pd.concat([data_train, data_val])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# GPU info
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

# CPU info
print(f"CPU RAM Free: {psutil.virtual_memory().available / 1024**3:.2f} GB")

# Lags used as predictors
regressor = XGBRegressor(
                            n_estimators=5000,
                            tree_method='gpu_hist',
                            gpu_id=0
                            )
forecaster = ForecasterAutoreg( regressor
               ,
                lags = duration_in_month
             )

lags_grid = [duration_in_day, duration_in_week, duration_in_month]

# Regressor hyperparameters search space
def search_space(trial):
    search_space  = {
        #'n_estimators'     : trial.suggest_int('n_estimators', 1000, 1100),
        'learning_rate'    : trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth'        : trial.suggest_int('max_depth', 10, 15),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1., 10),
        'max_features'     : trial.suggest_categorical('max_features', ['log2', 'sqrt'])
    }
    return search_space

results, frozen_trial = bayesian_search_forecaster(
                            forecaster            = forecaster,
                            y                     = data_train_val['y'],
                            exog                  = data_train_val[exog_vars],
                            lags_grid             = lags_grid,
                            search_space          = search_space,
                            steps                 = steps,
                            metric                = ['mean_absolute_percentage_error','mean_squared_error'],
                            refit                 = True,
                            initial_train_size    = len(data_train),
                            fixed_train_size      = False,
                            n_trials              = 10,
                            random_state          = 123,
                            return_best           = False,
                            n_jobs                = 'auto',
                            verbose               = True,
                            show_progress         = True,
                            engine                = 'optuna',
                            kwargs_create_study   = {},
                            kwargs_study_optimize = {},
                            gap  = duration_in_week
                        )

cwd = os.path.dirname(__file__)
regressor_name_to_save = str(regressor).split('(')[0]
path = os.path.join(cwd, f'./data/results_gpu_train_{regressor_name_to_save}.csv')
results.to_csv(path)