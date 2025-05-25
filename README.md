# 📊 StockPricePredictionLSTM 📊

This project is a Python software designed to predict future stock prices for a given ticker symbol using a LSTM neural network. It fetches historical stock data, preprocesses it, builds and trains an LSTM model, and generates a plot visualizing the historical data alongside the predicted future prices.

## Overview 🧐

The main goal of this software is to:

* Take a stock ticker symbol as input.
* Retrieve historical stock price data for the specified ticker.
* Normalize the historical closing prices.
* Create and compile an LSTM model for time series forecasting.
* Generate predictions for a specified number of future days (defaulting to 100).
* Plot the historical closing prices and the predicted future prices in a single figure, saved to the main directory.

## Technologies Used 💻

* **Python**
* **pandas**
* **NumPy**
* **scikit-learn**
* **TensorFlow**
* **Keras**
* **Matplotlib**

## Project Structure 📂

The repository contains the following files:

* `main.py`: The main script that takes the ticker and prediction days as (default) inputs, fetches data, trains the LSTM model, makes predictions, and generates the plot.
* `utils/`: A directory containing utility modules:
    * `Get_data.py`: Contains the function to retrieve historical stock data.
    * `Data_normalization.py`: Contains the function to normalize the stock price data.
    * `LSTM_creation.py`: Contains the function to create the LSTM model.
    * `Model_compilation.py`: Contains the function to compile the LSTM model.
    * `Prediction.py`: Contains the function to make future price predictions.
    * `Plot.py`: Contains the function to generate and save the plot.

## Customization

You can easily modify the script to predict different stocks or a different number of future days by changing the default values in the `main.py` script:

```python
# User inputs
ticker = "AAPL"  # Change to your desired ticker
days = 30    # Change to the number of days you want to predict
