# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary modules and functions
from ml_pipeline.utils import *
from ml_pipeline.train import *
import yfinance as yf
import numpy as np
from projectpro import model_snapshot, checkpoint

# Override Yahoo Finance's download method
yf.pdr_override()

# Load historical stock price data for AAPL
dataset = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

# Create a checkpoint for tracking progress
checkpoint('34db30')
print("Data Loaded")

# Set the start and end years for data splitting
tstart = 2016
tend = 2020

# Split the dataset into training and test sets
training_set, test_set = train_test_split(dataset, tstart, tend)

# Scale dataset values using Min-Max scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)

# Create overlapping window batches
n_steps = 1
features = 1
X_train, y_train = split_sequence(training_set_scaled, n_steps)

# Reshape X_train for model compatibility
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

# Train the RNN model and save it
model_rnn = train_rnn_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path="output/model_rnn.h5")
model_snapshot("34db30")

# Train the LSTM model and save it
model_lstm = train_lstm_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path="output/model_lstm.h5")

# Set the number of multivariate features
mv_features = 6

# Process and split multivariate data
X_train, y_train, X_test, y_test, mv_sc = process_and_split_multivariate_data(dataset, tstart, tend, mv_features)

# Train the multivariate LSTM model and save it
model_mv = train_multivariate_lstm(X_train, y_train, X_test, y_test, mv_features, mv_sc, save_model_path="output/model_mv_lstm.h5")
model_snapshot("34db30")
