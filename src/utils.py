import pandas as pd

class FeatureEngineering:
  "class to do feature engineering"
  def get_basic_temporal_features(self,
                                  df: pd.DataFrame)-> pd.DataFrame:
    data = df.copy()
    data["month"] = data.index.month
    data["dayofweek"] = data.index.dayofweek
    data["hour"] = data.index.hour
    return data[["month","dayofweek","hour"]]

  def get_rolling_data(self,
                      df: pd.DataFrame,
                      parameter: str,
                      duration: int,
                      duration_unit: str,
                      agg_method: str)-> pd.DataFrame:

    if duration_unit == "D":
      total_duration = duration * 24

    new_rolling_parameter_name   = f"{parameter}_rolling_{agg_method}_{duration}_{duration_unit}"
    new_rolling_parameter_f      = df[parameter].rolling(total_duration)
    new_rolling_parameter_series = getattr(new_rolling_parameter_f, agg_method)()
    new_rolling_parameter_series.name = new_rolling_parameter_name
    return new_rolling_parameter_series.to_frame()

class DataCleaningService:

  def normalize_data(self,
                     df: pd.DataFrame,
                     start_date: str,
                     end_date  : str):

    #real_start_date = str(pd.to_datetime(start_date, format = "%Y-%m-%d") - pd.Timedelta(offset_rolling_feature_in_h, "H"))[:10]

    #date_range = f"{start_date}:{end_date}"
    df.index = pd.to_datetime(df["Datetime"], utc = True)
    df = df.drop(columns = ["Datetime", "Resolution code"])
    df = df.sort_index().loc[start_date:end_date]
    return df.apply(lambda x : x.astype(float))

  def check_missing_date(self,
                         df: pd.DataFrame):
    missing_date = pd.date_range(start = df.index[0],
                                 end   = df.index[-1] ).difference(df.index)
    if len(missing_date) > 0:
      return missing_date
    else:
      return print("no missing dates detected!")
