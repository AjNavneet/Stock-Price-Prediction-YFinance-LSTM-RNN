import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Function to split data into training and testing sets
def train_test_split(dataset, tstart, tend, columns=['High']):
    train = dataset.loc[f"{tstart}":f"{tend}", columns].values
    test = dataset.loc[f"{tend+1}":, columns].values
    return train, test

# Function to split a sequence into input-output pairs
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to calculate and print RMSE (Root Mean Squared Error)
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))

# Function to process and split multivariate data
def process_and_split_multivariate_data(dataset, tstart, tend, mv_features):
    multi_variate_df = dataset.copy()

    # Technical Indicators
    multi_variate_df['RSI'] = ta.rsi(multi_variate_df.Close, length=15)
    multi_variate_df['EMAF'] = ta.ema(multi_variate_df.Close, length=20)
    multi_variate_df['EMAM'] = ta.ema(multi_variate_df.Close, length=100)
    multi_variate_df['EMAS'] = ta.ema(multi_variate_df.Close, length=150)

    # Target Variable
    multi_variate_df['Target'] = multi_variate_df['Adj Close'] - dataset.Open
    multi_variate_df['Target'] = multi_variate_df['Target'].shift(-1)
    multi_variate_df.dropna(inplace=True)

    # Drop unnecessary columns
    multi_variate_df.drop(['Volume', 'Close'], axis=1, inplace=True)

    # Plotting
    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'RSI']].plot(figsize=(16, 4), legend=True)

    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'EMAF', 'EMAM', 'EMAS']].plot(figsize=(16, 4), legend=True)

    feat_columns = ['Open', 'High', 'RSI', 'EMAF', 'EMAM', 'EMAS']
    label_col = ['Target']

    # Splitting train and test data
    mv_training_set, mv_test_set = train_test_split(multi_variate_df, tstart, tend, feat_columns + label_col)

    X_train = mv_training_set[:, :-1]
    y_train = mv_training_set[:, -1]

    X_test = mv_test_set[:, :-1]
    y_test = mv_test_set[:, -1]

    # Scaling Data
    mv_sc = MinMaxScaler(feature_range=(0, 1))
    X_train = mv_sc.fit_transform(X_train).reshape(-1, 1, mv_features)
    X_test = mv_sc.transform(X_test).reshape(-1, 1, mv_features)

    return X_train, y_train, X_test, y_test, mv_sc
