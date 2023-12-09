# Stock Price Prediction Using Yfinance, LSTM and RNN

### Business Context

Accurate stock price prediction is of paramount importance in financial markets, influencing investment decisions, risk management, and portfolio optimization. This project focuses on implementing recurrent neural networks (RNNs) and long short-term memory (LSTM) networks for stock price prediction, offering valuable insights into the intersection of deep learning and financial forecasting.

---

### Practical Application

Accurate stock price predictions are a game-changer for a wide range of stakeholders, including investors, traders, and financial analysts. These predictions enable investors to make informed decisions, identify investment opportunities, and adapt their portfolios in response to anticipated price movements.

---

### Challenges and Limitations

Stock price prediction is a challenging task due to the intricate nature of financial markets. This project acknowledges the presence of market volatility, external events, and data noise as factors that can affect prediction accuracy. It highlights the inherent limitations in forecasting financial market behavior.

---

### Objective

The project aims to achieve the following outcomes:

- Enhanced forecasting accuracy of stock prices.
- RNN and LSTM networks.
- Insights into how these models capture temporal dependencies in time series data.

---

## Data Description

The dataset used for this project comprises historical stock prices of Apple Inc. (AAPL), sourced from Yahoo Finance's API. It provides daily records of open, close, high, low prices, and trading volume for each trading day.

---

### Tech Stack

- Language: `Python`
- Libraries: `Keras`, `TensorFlow`, `Statsmodels`, `NumPy`, `Pandas`, `yfinance`, `pandas-datareader`, `pandas_ta`

---

### Approach

1. Understanding neural networks basics.
2. Loading time series data from Yahoo Finance.
3. Data preprocessing, including scaling, normalization, and window creation.
4. Building and training RNN models.
5. Model evaluation and sequence generation.
6. Extending the analysis to Long Short-Term Memory (LSTM) networks.
7. Incorporating technical indicators and multivariate inputs for more accurate predictions.

---

### Modular Code Structure

- **lib**: A reference folder containing original IPython notebooks.
- **ml_pipeline**: A folder with Python functions, where `engine.py` calls these functions to execute the project steps.
- **output**: Contains saved models produced by `engine.py`.
- **requirements.txt**: Lists all required libraries and their versions for installation.

---


