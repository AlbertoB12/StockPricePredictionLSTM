"""
# Util to normalize data from dataframe
"""

### Imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np

### Functions
## data_normalization(df)
# Function to normalize data to be processed by model to be trained
# Gets pd-dataframe with ticker data
# Returns x and y values for training
def data_normalization(df):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Range between 0 and 1
    scaled_data = scaler.fit_transform(df)
    # Prepare the data with past 100 days for LSTM input
    x_input = []
    y_input = []
    for i in range(100, scaled_data.shape[0]):
        x_input.append(scaled_data[i - 100: i])
        y_input.append(scaled_data[i, 0])
    x_input, y_input = np.array(x_input), np.array(y_input)
    return x_input, y_input, scaler
