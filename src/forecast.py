from sklearn.metrics import mean_absolute_percentage_error
from skforecast import model_selection
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import pandas as pd

class BacktestingForecaster():
    def __init__(
            self, 
            forecaster: ForecasterAutoreg,
            y: pd.DataFrame,
            steps: int,
            exog: pd.DataFrame,
            initial_train_size: int,
            metric,
            refit: bool,
            ) -> None:
        self.forecaster = forecaster
        self.y = y
        self.steps = steps
        self.exog = exog
        self.initial_train_size = initial_train_size
        self.metric = metric
        self.refit = refit

    def backtesting_forecaster(self):
        return model_selection.backtesting_forecaster(
            forecaster= self.forecaster,
            y = self.y,
            steps = self.steps,
            exog= self.exog,
            initial_train_size=self.initial_train_size,
            metric=self.metric,
            refit = self.refit,
            n_jobs = "auto",
            verbose = True,
            show_progress = True,
        )
    def _create_backtesting_folds(self):
        """ github code inputs mapping
        folds = _create_backtesting_folds(
                data                  = y,
                window_size           = window_size,
                initial_train_size    = initial_train_size,
                test_size             = steps,
                externally_fitted     = externally_fitted,
                refit                 = refit,
                fixed_train_size      = fixed_train_size,
                gap                   = gap,
                skip_folds            = skip_folds,
                allow_incomplete_fold = allow_incomplete_fold,
                return_all_indexes    = False,
                differentiation       = differentiation,
                verbose               = verbose
                forecaster.window_size_diff
        """
        return model_selection._create_backtesting_folds(  
            data=self.y,
            window_size= self.forecaster.window_size_diff,
            initial_train_size=self.initial_train_size,
            test_size=self.steps,
            refit=self.refit,
            verbose=False,
        )
    
    def get_backtesting_folds(self,
                              backtest_indexes: list,
                              backtest_result_df: pd.DataFrame,
                              regressor_name) -> list[pd.DataFrame]:
    
        df = pd.DataFrame([[n[0], n[3]] for n in backtest_indexes]) 

        index_to_suffix_dict = {0 : "_start",
                                1 : "_end"}

        def map_training_validation_indexes(df: pd.DataFrame, index_to_suffix_dict: dict) -> pd.DataFrame:
            df = pd.concat([pd.DataFrame
                                (df[index].to_list(), 
                                    columns=['training'  + index_to_suffix_dict[index],
                                            'validation'+ index_to_suffix_dict[index]]) \
                                            for index in index_to_suffix_dict], axis = 1)
            return df

        def readjust_end_indexes(df: pd.DataFrame) -> pd.DataFrame:
            df[["training_end", "validation_end"]] = df[["training_end", "validation_end"]].apply(lambda x : (x - 1))
            return df

        df_train_val_index_transformed = (df.pipe(map_training_validation_indexes, index_to_suffix_dict)
                                        .pipe(readjust_end_indexes))

        def get_training_validation_folds(df: pd.DataFrame, df_input: pd.DataFrame, df_pred: pd.DataFrame, folds_type: str)-> list:
            columns = df.filter(regex = folds_type).columns
            column_start, column_end = [col for col in columns if "start" in col], [col for col in columns if "end" in col]
            folds_list = []
            for fold in df.index:
                validation_start = df[column_start].loc[fold].values[0]
                validation_end   = df[column_end].loc[fold].values[0]
                df_input_ = df_input.iloc[validation_start:validation_end]
                date_index = df_input_.index
                df_pred_ = df_pred.loc[date_index]
                validation_fold = pd.concat([df_input_, df_pred_], axis = 1)
                folds_list.append(validation_fold)
            return folds_list 
        
        return get_training_validation_folds(
            df= df_train_val_index_transformed,
            df_input= self.y,
            df_pred= backtest_result_df[[regressor_name]],
            folds_type="validation")

        