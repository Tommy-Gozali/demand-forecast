from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np
import math

minute_in_hour = 60
hour_in_day = 24
day_in_week = 7
week_in_month = 4

class CreateExperiment:
    """class to create experiment"""
    global minute_in_hour, hour_in_day, day_in_week, week_in_month


    def __init__(self,
                data: pd.DataFrame,
                frequency_magnitude_input: int,
                frequency_unit_input: str, #m = minutes, H = hour
                frequency_magnitude_output: int,
                frequency_unit_output: str, #m = minutes, H = hour
                forecast_type: str,
                train_size: float,
                metric: str = "mean_absolute_percentage_error"):

        self.data = data
        self.frequency_magnitude_input = frequency_magnitude_input
        self.frequency_unit_input = frequency_unit_input
        self.frequency_magnitude_output = frequency_magnitude_output
        self.frequency_unit_output = frequency_unit_output
        self.forecast_type  = forecast_type
        self.train_size = train_size
        self.metric = metric

    @property
    def frequency_unit_input(self):
        return self._frequency_unit_input

    @property
    def frequency_unit_output(self):
        return self._frequency_unit_output


    @frequency_unit_input.setter
    def frequency_unit_input(self, value: str) -> str:

    allowed_values = ["m", "H"]
    if value in allowed_values:
        self._frequency_unit_input = value
    else:
        raise ValueError("Only 'm' and 'H' are allowed")

    @frequency_unit_output.setter
    def frequency_unit_output(self, value: str) -> str:

    allowed_values = ["m", "H"]
    if value in allowed_values:
        self._frequency_unit_output = value
    else:
        raise ValueError("Only 'm' and 'H' are allowed")

    @property
    def frequency_magnitude_input(self):
    return self._frequency_magnitude_input

    @property
    def frequency_magnitude_output(self):
        return self._frequency_magnitude_output

    @frequency_magnitude_input.setter
    def frequency_magnitude_input(self, value: int) -> int:

    allowed_values = [1, 15, 30]
    if value in allowed_values:
        self._frequency_magnitude_input = value
    else:
        raise ValueError("Only 1, 15, 30 are allowed")

    @frequency_magnitude_output.setter
    def frequency_magnitude_output(self, value: int) -> int:

    allowed_values = [1, 15, 30]
    if value in allowed_values:
        self._frequency_magnitude_output = value
    else:
        raise ValueError("Only 1, 15, 30 are allowed")

    """
    @property
    def frequency_input_output_validator(self):
    #check frequency unit first

    #only option that is not possible
    if self.frequency_unit_input == "D" and self.frequency_unit_output == "H":
        raise ValueError("Output have frequency unit is higher than input")

    elif self.frequency_unit_input == "H" and self.frequency_unit_output == "H":
        if self.frequency_unit_input < self.frequency_unit_output:
        pass
        else:
        raise ValueError("Output have frequency unit is higher than input")

    else:

        if self.frequency_unit_output == self.frequency_unit_input:


        elif (self.frequency_magnitude_input > self.frequency_magnitude_output):
        :
    """

    @property
    def frequency_converter(self):
    if self.frequency_magnitude_input == "m":
        self.frequency_magnitude_input = "min"
    elif self.frequency_magnitude_output == "m":
        self.frequency_magnitude_output = "min"

    freq_dest = str(self.frequency_magnitude_output) + self.frequency_unit_output
    return self.data.resample(freq_dest).mean()

    @property
    def basic_durations(self) -> float:
    if self.frequency_unit_output == "H" and self.frequency_magnitude_output == 1:
        durations_in_day_ = self.frequency_magnitude_output * hour_in_day

    elif self.frequency_unit_output == "m" and self.frequency_magnitude_output in [15,30]:
        durations_in_day_ = (minute_in_hour/self.frequency_magnitude_output) * hour_in_day

    else:
        raise ValueError("Mismatch frequency_unit_input and frequency_magnitude_input")

    durations_in_week_  = durations_in_day_  * day_in_week
    durations_in_month_ = durations_in_week_ * week_in_month

    return durations_in_day_, durations_in_week_, durations_in_month_

    @property
    def steps(self) -> int:
    if self.forecast_type == "day_ahead":
        return self.basic_durations[0]
    elif self.forecast_type == "week_ahead":
        return self.basic_durations[1]

    def create_train_test_validation_split(self) -> list[pd.DataFrame]:
    df_train, df_test = train_test_split(self.frequency_converter, test_size= (1-self.train_size), shuffle=False)
    df_test , df_val  = train_test_split(df_test, test_size=0.5, shuffle=False)
    return df_train, df_test, df_val

    def get_last_index_round(self, indexes, index_):
    last_index_round_date = indexes[:index_][-1].round("D")
    last_index_round_int  = indexes.tolist().index(last_index_round_date)
    return last_index_round_date, last_index_round_int

    def get_input_data(self):
    return self.frequency_converter

    def create_train_test_validation_split_round(self) -> list[pd.DataFrame().index]:
    indexes = self.frequency_converter.index

    index_train = math.ceil(len(indexes) * self.train_size)
    index_train_end_date, index_train_end_date_int = self.get_last_index_round(indexes, index_train)

    #index_test = index_train_end_date_int + math.ceil(len(indexes[index_train_end_date_int:]) * 0.5)
    #index_test_end_date, index_test_end_date_int = self.get_last_index_round(indexes, index_test)
    index_val = index_train_end_date_int + math.ceil(len(indexes[index_train_end_date_int:]) * 0.5)
    index_val_end_date, index_val_end_date_int = self.get_last_index_round(indexes, index_val)

    train_indexes  = indexes[:index_train_end_date_int]
    val_indexes    = indexes[index_train_end_date_int:index_val_end_date_int]
    test_indexes   = indexes[index_val_end_date_int:]

    return train_indexes, val_indexes, test_indexes
