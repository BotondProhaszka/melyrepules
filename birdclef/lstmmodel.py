import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
import numpy as np
from sklearn.metrics import average_precision_score

def padded_cmap_tensorflow(y_true, y_pred, padding_factor=5):
    y_pred = tf.argmax(y_pred, axis=-1)
    def padded_cmap_np(y_true_np, y_pred_np):
        y_true_padded = np.pad(y_true_np, ((0, padding_factor), (0, 0)), constant_values=1)
        y_pred_padded = np.pad(y_pred_np, ((0, padding_factor), (0, 0)), constant_values=1)

        # Flatten the arrays
        y_true_flat = y_true_padded.flatten()
        y_pred_flat = y_pred_padded.flatten()

        return average_precision_score(y_true_flat.astype(int), y_pred_flat, average="macro")
    
    # Use tf.py_function to execute the NumPy operation eagerly
    result_tensor = tf.py_function(padded_cmap_np, [y_true, y_pred], tf.float32)
    return result_tensor

class LSTMModel():
    
    def __init__ (self, num_labels, input_shape, loss = 'mean_squared_error', optimizer = 'adam', metrics = 'accuracy'):

        self.num_labels = num_labels
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer

        self.model = Sequential([
            layers.LSTM(4, input_shape = self.input_shape, activation='relu'),
            layers.Dense(self.num_labels)
        ])

        self.model.summary()
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=metrics,
        )

    def train(self, data, val_data, epochs = 100, checkpoint_filepath = 'saved_model/1', batch_size = 30, verbose = 1):

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        history = self.model.fit(
            data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[model_checkpoint_callback]
        )

        self.model.save_weights(checkpoint_filepath)
        self.model.save(checkpoint_filepath)

        return history
        
    def predict(self, test_data):
        y_pred = self.model.predict(test_data)
        y_pred = tf.argmax(y_pred, axis=1)
        return y_pred
    
    def evaluate(self, test_data):
        self.model.evaluate(test_data, return_dict=True)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

