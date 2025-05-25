"""
### InsightLyze - Stock Price Prediction ###
# Main script
# Gets ticker name from user (str)
# Returns a plot figure with prediction for ticker, saved in main directory
# Script gets historical data from ticker, data is normalized, an LSTM model is created and compiled, a prediction is made
# Do not modify code
# Author and Copyright: InsightLyze
"""

### Imports
from utils import Get_data, Data_normalization, LSTM_creation, Model_compilation, Prediction, Plot

### Code
# User inputs
ticker = "TSLA"  # Desired ticker
days = 100  # Desired days to predict ahead

# Get data (close prices for training model) as pd-dataframe
data = Get_data.get_data(ticker)  # from: .\utils\Get_data.py
# Extract the 'Close' prices and reshape for scaling
close_data = data['Close'].values.reshape(-1, 1)
# Normalize the data
x_input, y_input, scaler = Data_normalization.data_normalization(close_data)
# Model creation (LSTM)
model = LSTM_creation.create_LSTM(x_input)
# Model compilation
Model_compilation.compile(model, x_input, y_input, 'adam', 'mean_squared_error', 10)
# Prediction
predictions = Prediction.predict(model, days, scaler, close_data)
# Plot
Plot.plot(close_data, predictions, ticker, 'Time', 'Price')
