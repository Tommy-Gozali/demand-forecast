import requests
import json
import pandas as pd

class OpenMeteoApi:
  "class to get data from open-meteo"
  def __init__(self, longitude, latitude):
    self.longitude = longitude
    self.latitude  = latitude

  def initiate_parameters(self,
                          start_date: str,
                          end_date  : str,
                          parameters: list
                          ) -> str:
    if isinstance(parameters, list):
      if len(parameters) == 0:
        print("missing parameter in the list")
      else:
        parameters_str = ",".join(parameters)

    url = f"latitude={self.latitude}" + "&" + f"longitude={self.longitude}" + "&" +\
          f"hourly={parameters_str}"  + "&" +\
          f"start_date={start_date}"  + "&" + f"end_date={end_date}"
    return url

  def response_2_dataframe(self,
                           url: str) -> pd.DataFrame:

    response = requests.get(url).content
    data_ = pd.DataFrame(json.loads(response)["hourly"]).set_index("time")
    data_.index = pd.to_datetime(data_.index, utc = True)
    return data_


  def get_historical_actual_data(self,
                                 parameters_ : str) -> pd.DataFrame:

    base_url = "https://archive-api.open-meteo.com/v1/archive?"
    mod_url  = base_url + parameters_
    data_ = self.response_2_dataframe(mod_url)
    return data_

  def get_historical_forecast_data(self,
                                   parameters_ : str) -> pd.DataFrame:

    base_url = "https://previous-runs-api.open-meteo.com/v1/forecast?"
    mod_url  = base_url + parameters_
    data_ = self.response_2_dataframe(mod_url)
    return data_

  def get_current_forecast_data(self,
                                   parameters_ : str) -> pd.DataFrame:

    base_url = "https://api.open-meteo.com/v1/forecast?"
    mod_url  = base_url + parameters_
    data_ = self.response_2_dataframe(mod_url)
    return data_
  
