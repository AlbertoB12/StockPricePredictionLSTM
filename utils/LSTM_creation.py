"""
# Util to create LSTM model for training and predicting stock prices
"""

### Imports
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential

### Functions
## create_LSTM(x_train)
# Function to create LSTM model for training and predicting stock prices
# Gets x train data (df)
# Returns tf.keras LSTM model
def create_LSTM(x_input):
    model = Sequential([
        # First LSTM layer with Batch Normalization and Dropout
        LSTM(50, activation='relu', return_sequences=True, input_shape=(x_input.shape[1], 1)),
        Dropout(0.2),
        # Second LSTM layer with Batch Normalization and Dropout
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.3),
        # Third LSTM layer with Batch Normalization and Dropout
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.4),
        # Fourth LSTM layer with Batch Normalization and Dropout
        LSTM(120, activation='relu'),
        Dropout(0.5),
        # Dense output layer with linear activation
        Dense(1)
    ])
    return model
