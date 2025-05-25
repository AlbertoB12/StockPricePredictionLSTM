"""
# Util to compile and train model
"""

### Imports
import tensorflow as tf

### Functions
## def compile(model, x_train, y_train, optimizer, loss, epochs)
# Function to compile and train model
# Gets model (tf.keras model), x_train (df), y_train (df), optimizer, loss, epochs (int)
# No return
def compile(model, x_input, y_input, optimizer, loss, epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_input, y_input, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
