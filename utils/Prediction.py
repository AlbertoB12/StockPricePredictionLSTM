"""
# Util to make a prediction based on the trained model
"""

### Imports
import numpy as np

### Functions
## def predict(model, x_test)
# Main function to make a prediction based on the trained model and plot it
# Gets model (tf.keras model), days to be predicted(int), scaler, data (df)
# No return, a png-file with figure is saved in the same directory with the name 'Prediction' + ticker name
def predict(model, days, scaler, close_data):
    # Predict the next x days
    predicted_values = []
    current_input = scaler.transform(close_data[-100:])  # Use the last 100 days for starting the prediction loop
    # Predict x days into the future
    for _ in range(days):
        next_pred = model.predict(current_input)
        predicted_values.append(next_pred[0, 0])  # Add the predicted value to the list
        # Reshape next_pred to be compatible for appending
        next_pred_reshaped = next_pred.reshape(1, 1, 1)  # Reshape next_pred to (1, 1, 1)
        # Shift the window: drop the first day and add the prediction as the new last day
        current_input = np.append(current_input[:, 1:, :], next_pred_reshaped, axis=1)  # Append with proper shape
    # Rescale predicted values to the original price scale
    predicted_values = np.array(predicted_values).reshape(-1, 1)
    predicted_values = predicted_values * (1 / scaler.scale_[0])  # Scale back using scale factor
    return predicted_values
