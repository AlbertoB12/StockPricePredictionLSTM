"""
# Util to get data from one ticker from the yf-server
"""

### Imports
import yfinance as yf
import pandas as pd

### Functions
## get_data(ticker)
# Get data from one ticker from the yf-server
# Gets str (name of ticker)
# Returns pd-dataframe with close-price-data for training
def get_data(ticker):
    data = yf.download(ticker, period="max")
    # Drop multi-index if it exists (like ticker level)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data  # Return full DataFrame with OHLCV columns
