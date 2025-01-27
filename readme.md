# Stock Price Prediction Using Yfinance, LSTM, and RNN

## Business Context

Accurate stock price prediction plays a pivotal role in financial markets, influencing investment decisions, risk management, and portfolio optimization. This project leverages **recurrent neural networks (RNNs)** and **long short-term memory (LSTM)** networks for stock price prediction, showcasing the application of deep learning techniques in financial forecasting.

---

## Practical Application

Stock price prediction benefits a wide range of stakeholders, including:

- **Investors**: Make informed investment decisions and identify opportunities.
- **Traders**: Adapt strategies to anticipated price movements.
- **Financial Analysts**: Analyze market trends to provide actionable insights.

By accurately forecasting stock prices, stakeholders can manage risks and optimize portfolios more effectively.

---

## Challenges and Limitations

Stock price prediction is inherently challenging due to:

- **Market Volatility**: Unpredictable price fluctuations driven by external events.
- **Data Noise**: Irregularities and inconsistencies in financial data.
- **External Factors**: Global events, policy changes, and other variables impacting market behavior.

This project acknowledges these challenges and aims to highlight how RNNs and LSTMs address temporal dependencies in financial data.

---

## Objective

The project aims to achieve the following outcomes:

- Improved forecasting accuracy of stock prices.
- Effective use of **RNN** and **LSTM** networks.
- Insights into how deep learning models capture temporal dependencies in time series data.

---

## Data Description

The dataset consists of historical stock prices for **Apple Inc. (AAPL)**, sourced using the [Yahoo Finance API](https://pypi.org/project/yfinance/). Key features include:

- **Open**: Opening price of the stock.
- **Close**: Closing price of the stock.
- **High**: Highest price for the trading day.
- **Low**: Lowest price for the trading day.
- **Volume**: Number of shares traded.

---

## Tech Stack

- **Programming Language**: [Python](https://www.python.org/)
- **Libraries**:
  - [`Keras`](https://keras.io/) and [`TensorFlow`](https://www.tensorflow.org/) for building and training neural networks.
  - [`Statsmodels`](https://www.statsmodels.org/) for statistical analysis.
  - [`NumPy`](https://numpy.org/) and [`Pandas`](https://pandas.pydata.org/) for data manipulation.
  - [`yfinance`](https://pypi.org/project/yfinance/) for retrieving stock data.
  - [`pandas-datareader`](https://pandas-datareader.readthedocs.io/en/latest/) for accessing financial data.
  - [`pandas-ta`](https://github.com/twopirllc/pandas-ta) for technical analysis indicators.

---

## Approach

### 1. Neural Network Basics
- Understand the concepts of RNNs and LSTMs for handling sequential data.

### 2. Load Stock Data
- Retrieve historical stock price data using **Yahoo Finance API**.

### 3. Data Preprocessing
- Scale and normalize stock price data.
- Create sliding windows for time series analysis.

### 4. RNN Model Development
- Build and train recurrent neural networks for stock price prediction.

### 5. LSTM Model Development
- Extend the analysis by implementing LSTM networks to capture long-term dependencies.

### 6. Incorporate Technical Indicators
- Use additional features, such as technical analysis indicators, for improved predictions.

### 7. Model Evaluation
- Assess the performance of the RNN and LSTM models using appropriate metrics.
- Generate sequences for forecasting.

---

## Modular Code Structure

```plaintext
.
├── lib/                                   # Reference folder with original IPython notebooks.
├── ml_pipeline/                           # Folder with modular Python scripts.
│   ├── data_preparation.py                # Functions for data preprocessing.
│   ├── model_training.py                  # Functions for training RNN and LSTM models.
│   ├── evaluation.py                      # Functions for evaluating model performance.
│   ├── engine.py                          # Main script to execute the pipeline.
├── output/                                # Stores saved models and evaluation results.
├── requirements.txt                       # Lists all required libraries and their versions.
└── README.md                              # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

Execute the pipeline by running the `engine.py` script:

```bash
python ml_pipeline/engine.py
```

### 4. Explore Results

- Review model outputs and predictions in the `output/` folder.
- Analyze the evaluation metrics for model performance.

---

## Results

- **Forecasting Accuracy**:
  - RNN and LSTM models effectively captured temporal patterns in stock prices.
- **Technical Indicator Integration**:
  - Enhanced predictions by incorporating additional features.
- **Deep Learning Insights**:
  - Demonstrated how RNNs and LSTMs model sequential dependencies in financial data.



---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Abhinav Navneet
- **Email**: mailme.AbhinavN@gmail.com
- **GitHub**: [AjNavneet](https://github.com/AjNavneet)

---

## Acknowledgments

Special thanks to:

- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for providing robust deep learning frameworks.
- [Yahoo Finance API](https://pypi.org/project/yfinance/) for stock data retrieval.
- [Statsmodels](https://www.statsmodels.org/) and [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical analysis.
- The Python open-source community for their excellent tools and libraries.

---

