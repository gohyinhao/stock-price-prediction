import pandas as pd
import numpy as np

# class to preprocess data given CSV file containing stock/index prices.
# breaks up data into sets of {@param: seq_len} prices and the next day price (target output).


class DataProcessing:
    def __init__(self, data, percentage_as_training, n_day_forecast):
        self.data = data
        self.percentage_as_training = percentage_as_training
        self.n_day_forecast = n_day_forecast
        self.index = int(self.percentage_as_training * len(self.data))
        self.training_set = self.data[0: self.index]
        self.validation_set = self.data[self.index:]
        self.training_input = []
        self.training_output = []
        self.validation_input = []
        self.validation_output = []

    # split data set into training set based on percentage_as_training
    def generate_training_set(self, seq_len):
        """
        Generates training data
        :param seq_len: length of window
        :return: X_training and Y_training
        """

        # represents zero-based column number in CSV file
        COLUMN_NUMBER = 0

        for i in range((len(self.training_set)//seq_len)*seq_len - seq_len - self.n_day_forecast):
            x = np.array(self.training_set.iloc[i: i + seq_len, COLUMN_NUMBER])
            y = np.array(
                [self.training_set.iloc[i + seq_len + self.n_day_forecast, COLUMN_NUMBER]], np.float64)
            self.training_input.append(x)
            self.training_output.append(y)
        self.X_training = np.array(self.training_input)
        self.Y_training = np.array(self.training_output)

    # split data set into validation set based on percentage_as_training
    def generate_validation_set(self, seq_len):
        """
        Generates validation data
        :param seq_len: Length of window
        :return: X_test and Y_test
        """

        # represents zero-based column number in CSV file
        COLUMN_NUMBER = 0

        for i in range((len(self.validation_set)//seq_len)*seq_len - seq_len - self.n_day_forecast):
            x = np.array(
                self.validation_set.iloc[i: i + seq_len, COLUMN_NUMBER])
            y = np.array(
                [self.validation_set.iloc[i + seq_len + self.n_day_forecast, COLUMN_NUMBER]], np.float64)
            self.validation_input.append(x)
            self.validation_output.append(y)
        self.X_validation = np.array(self.validation_input)
        self.Y_validation = np.array(self.validation_output)
