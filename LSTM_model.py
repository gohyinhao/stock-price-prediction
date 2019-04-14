import pandas as pd
import numpy as np
import tensorflow as tf
from preprocessing import DataProcessing

# optimizable variables
PERCENTAGE_OF_DATA_FOR_TRAINING = 0.8
SEQ_LENGTH = 10

data_processor = DataProcessing(
    "data\VTI.csv", PERCENTAGE_OF_DATA_FOR_TRAINING)
data_processor.generate_validation_set(SEQ_LENGTH)
data_processor.generate_training_set(SEQ_LENGTH)

# typically normalize by substracting average and dividing by standard deviation
# but we want to use the output and check against live trade data thus not entirely feasible
# hence simply divide by a constant to ensure weights in network do not become too large
NORMALIZATION_CONSTANT = 200

# "normalization" of training data
X_training = data_processor.X_training.reshape(
    len(data_processor.X_training), SEQ_LENGTH, 1) / NORMALIZATION_CONSTANT
Y_training = data_processor.Y_training / NORMALIZATION_CONSTANT

# "normalization" of validation data
X_validation = data_processor.X_validation.reshape(
    len(data_processor.X_validation), SEQ_LENGTH, 1) / NORMALIZATION_CONSTANT
Y_validation = data_processor.Y_validation / NORMALIZATION_CONSTANT

# RNN with 2 hidden layers
# consider changing number of hidden layers as well as number of nodes
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(
    SEQ_LENGTH, 1), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

# adam optimizer used in lieu of traditional stochastic gradient descent.
# consider possibility of looking at other optimizers
model.compile(optimizer="adam", loss="mean_squared_error")

# fit model with our training data
# vary epochs to optimize?
model.fit(X_training, Y_training, epochs=50)

# test model against validation data set
print(model.evaluate(X_validation, Y_validation))

# If instead of a full backtest, you just want to see how accurate the model is for a particular prediction, run this:
# data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
# stock = data["Adj Close"]
# X_predict = np.array(stock).reshape((1, 10)) / NORMALIZATION_CONSTANT
# print(model.predict(X_predict) * NORMALIZATION_CONSTANT)
