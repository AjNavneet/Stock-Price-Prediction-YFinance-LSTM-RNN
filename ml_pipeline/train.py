# Import necessary libraries and modules
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr

# Import Keras layers and models
from keras.layers import LSTM, SimpleRNN
from keras.models import Sequential

# Import custom utility functions
from .utils import split_sequence

# Set the number of time steps and features
n_steps = 1
features = 1

# Define a function for generating a sequence of future predictions
def sequence_generation(dataset: pd.DataFrame, sc: MinMaxScaler, model: Sequential, steps_future: int, test_set):
    high_dataset = dataset.iloc[len(dataset) - len(test_set) - n_steps:]["High"]
    high_dataset = sc.transform(high_dataset.values.reshape(-1, 1))
    inputs = high_dataset[:n_steps]

    for _ in range(steps_future):
        curr_pred = model.predict(inputs[-n_steps:].reshape(-1, n_steps, features), verbose=0)
        inputs = np.append(inputs, curr_pred, axis=0)

    return sc.inverse_transform(inputs[n_steps:])

# Define a function for training an RNN model
def train_rnn_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path=None):
    model = Sequential()
    model.add(SimpleRNN(units=125, input_shape=(n_steps, features))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    # Scaling
    inputs = sc.transform(test_set.reshape(-1, 1))

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    # reshape
    X_test = X_test.reshape(-1, n_steps, features)

    # Prediction
    predicted_stock_price = model.predict(X_test)
    # Inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    # Call sequence_generation function
    
    results = sequence_generation(dataset, sc, model, steps_in_future, test_set)
    print("Generated sequence of future predictions:")
    print(results)
    
    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")
    
    return model

# Define a function for training an LSTM model
def train_lstm_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path=None):
    model = Sequential()
    model.add(LSTM(units=125, input_shape=(n_steps, features))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    # Scaling
    inputs = sc.transform(test_set.reshape(-1, 1))

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    # reshape
    X_test = X_test.reshape(-1, n_steps, features)

    # Prediction
    predicted_stock_price = model.predict(X_test)
    # Inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    # Call sequence_generation function
    
    results = sequence_generation(dataset, sc, model, steps_in_future, test_set)
    print("Generated sequence of future predictions:")
    print(results)
    
    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")
    
    return model

# Define a function for training a multivariate LSTM model
def train_multivariate_lstm(X_train, y_train, X_test, y_test, mv_features, mv_sc, save_model_path=None):
    model_mv = Sequential()
    model_mv.add(LSTM(units=125, input_shape=(1, mv_features))
    model_mv.add(Dense(units=1)

    # Compiling the model
    model_mv.compile(optimizer="RMSprop", loss="mse")

    history = model_mv.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    predictions = model_mv.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    if save_model_path:
        model_mv.save(save_model_path)
        print("Model saved successfully.")

    return model_mv
