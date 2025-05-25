"""
# Util to plot an object
"""

### Imports
import matplotlib.pyplot as plt

### Functions
## def plot(object)
# Function to plot an object
# Gets object (df)
# Returns: Graph as PNG-file saved in main directory
def plot(close_data, predicted_values, ticker, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(close_data[-100:], 'b', label="Last 100 Actual Prices")  # Last 100 actual data points
    plt.plot(range(100, (len(predicted_values)+100)), predicted_values, 'r', label=f"Next {len(predicted_values)} Predicted Days")  # Next x predictions
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.title(f'Prediction of {ticker} Stock Price for Next 100 Days')
    plt.show()
