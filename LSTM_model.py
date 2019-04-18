import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import quandl as quandl
import keras
from preprocessing import DataProcessing

# live data extraction
TICKER = "WGC/GOLD_DAILY_USD"
data = quandl.get("WGC/GOLD_DAILY_USD", authtoken="FUw5PPY7UX5AW-svaskC")
data = data[37:]  # before index 37, prices were not daily

# optimizable variables
PERCENTAGE_OF_DATA_FOR_TRAINING = 0.8
SEQ_LENGTH = 30
N_DAY_FORECAST = 7

data_processor = DataProcessing(
    data, PERCENTAGE_OF_DATA_FOR_TRAINING, N_DAY_FORECAST)
data_processor.generate_validation_set(SEQ_LENGTH)
data_processor.generate_training_set(SEQ_LENGTH)

# typically normalize by substracting average and dividing by standard deviation
# but we want to use the output and check against live trade data thus not entirely feasible
# hence simply divide by a constant to ensure weights in network do not become too large
NORMALIZATION_CONSTANT = 500

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
NUM_OF_NODES = 20
NUM_OF_OUTPUT = 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(NUM_OF_NODES, input_shape=(
    SEQ_LENGTH, 1), return_sequences=True))
model.add(tf.keras.layers.LSTM(NUM_OF_NODES))
# uses Rectified Linear Unit (ReLU) activation function
model.add(tf.keras.layers.Dense(NUM_OF_OUTPUT, activation=tf.nn.leaky_relu))


# adam optimizer used in lieu of traditional stochastic gradient descent.
# consider possibility of looking at other optimizers
model.compile(optimizer="adam", loss="mean_squared_error")

# fit model with our training data
es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
model.fit(X_training, Y_training, epochs=500, callbacks=[es])

# test model against validation data set
print(model.evaluate(X_validation, Y_validation))

# get un-normalized data
X_actual = X_validation * NORMALIZATION_CONSTANT
Y_actual = Y_validation * NORMALIZATION_CONSTANT
Y_prediction = model.predict(X_validation) * NORMALIZATION_CONSTANT

# plot prediction vs. actual
plt.plot(Y_prediction, color='red', label='Predicted Gold Price')
plt.plot(Y_actual, color='blue', label='Actual Gold Price')
plt.title('Gold 7-day Price Forecast')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
tf.keras.backend.clear_session()

# getting next prediction
"""
last_price = np.array(
    data.iloc[len(data) - 10: len(data), 0]).reshape((1, 10)) / NORMALIZATION_CONSTANT
predicted_price = model.predict(last_price) * NORMALIZATION_CONSTANT
"""

# getting overall accuracy
accuracy = 0
for i in range(len(Y_prediction)):
    difference = abs(Y_prediction[i] - Y_actual[i]) / Y_actual[i]
    accuracy = accuracy + difference
accuracy = accuracy / len(Y_prediction) * 100
print("accuracy is", accuracy)

# accuracy excluding spike
partial_accuracy = 0
for i in range(500, len(Y_prediction)):
    difference = abs(Y_prediction[i] - Y_actual[i]) / Y_actual[i]
    partial_accuracy = partial_accuracy + difference
partial_accuracy = partial_accuracy / (len(Y_prediction)-500) * 100
print("partial accuracy is", partial_accuracy)

# If instead of a full backtest, you just want to see how accurate the model is for a particular prediction, run this:
# data = pdr.get_data_yahoo("AAPL", "2017-12-19", "2018-01-03")
# stock = data["Adj Close"]
# X_predict = np.array(stock).reshape((1, 10)) / NORMALIZATION_CONSTANT
# print(model.predict(X_predict) * NORMALIZATION_CONSTANT)
