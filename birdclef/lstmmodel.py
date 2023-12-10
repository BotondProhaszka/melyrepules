import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
import numpy as np
from sklearn.metrics import average_precision_score

# Custom TensorFlow function for calculating mean average precision with padding
def padded_cmap_tensorflow(y_true, y_pred, padding_factor=5):
    """
    Calculates mean average precision with padding using NumPy implementation.

    Parameters:
    - y_true (tf.Tensor): True labels.
    - y_pred (tf.Tensor): Predicted labels.
    - padding_factor (int): Padding factor for arrays (default is 5).

    Returns:
    - result_tensor (tf.Tensor): Resulting tensor with the mean average precision score.
    """
    # Convert predicted labels to class indices
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Numpy implementation of padded_cmap for eager execution
    def padded_cmap_np(y_true_np, y_pred_np):
        # Pad true and predicted labels
        y_true_padded = np.pad(y_true_np, ((0, padding_factor), (0, 0)), constant_values=1)
        y_pred_padded = np.pad(y_pred_np, ((0, padding_factor), (0, 0)), constant_values=1)

        # Flatten the arrays
        y_true_flat = y_true_padded.flatten()
        y_pred_flat = y_pred_padded.flatten()

        # Calculate mean average precision
        return average_precision_score(y_true_flat.astype(int), y_pred_flat, average="macro")
    
    # Use tf.py_function to execute the NumPy operation eagerly
    result_tensor = tf.py_function(padded_cmap_np, [y_true, y_pred], tf.float32)
    return result_tensor

# Class definition for LSTMModel
class LSTMModel():
    """
    LSTM Audio Classification Model.

    Attributes:
    - num_labels (int): Number of unique labels for classification.
    - input_shape (tuple): Shape of the input data.
    - loss (str): Loss function for model training (default is 'mean_squared_error').
    - optimizer (str): Optimizer for model training (default is 'adam').

    Methods:
    - __init__(self, num_labels, input_shape, loss='mean_squared_error', optimizer='adam', metrics='accuracy'): Initializes the LSTMModel object.
    - train(self, data, val_data, epochs=100, checkpoint_filepath='saved_model/1', batch_size=30, verbose=1): Trains the model on the provided training data.
    - predict(self, test_data): Generates predictions on the provided test data.
    - evaluate(self, test_data): Evaluates the model on the provided test data.
    - save(self, path): Saves the model at the specified path.
    - load(self, path): Loads a pre-trained model from the specified path.
    """
    
    def __init__(self, num_labels, input_shape, loss='mean_squared_error', optimizer='adam', metrics='accuracy', class_weights=None):
        """
        Initializes the LSTMModel object.

        Parameters:
        - num_labels (int): Number of unique labels for classification.
        - input_shape (tuple): Shape of the input data.
        - loss (str): Loss function for model training (default is 'mean_squared_error').
        - optimizer (str): Optimizer for model training (default is 'adam').
        - metrics (str): Evaluation metrics for model training (default is 'accuracy').
        - class_weights (dict): Class weights for imbalanced data (default is None).
        """
        # Initialize model parameters
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.class_weights = class_weights

        # Define LSTM model architecture
        self.model = Sequential([
            layers.LSTM(4, input_shape=self.input_shape, activation='relu'),
            layers.Dense(self.num_labels)
        ])

        # Display model summary
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def train(self, data, val_data, epochs=100, checkpoint_filepath='saved_model/1', batch_size=30, verbose=1):
        """
        Trains the model on the provided training data.

        Parameters:
        - data (tf.keras.utils.Sequence): Training data generator.
        - val_data (tf.keras.utils.Sequence): Validation data generator.
        - epochs (int): Number of training epochs (default is 100).
        - checkpoint_filepath (str): Path to save model checkpoints (default is 'saved_model/1').
        - batch_size (int): Batch size for training (default is 30).
        - verbose (int): Verbosity mode (default is 1).

        Returns:
        - history: Training history.
        """
        # Define model checkpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        # Train the model
        history = self.model.fit(
            data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[model_checkpoint_callback],
            class_weight=self.class_weights,
            batch_size=batch_size,
            verbose=verbose
        )

        # Save model weights and the entire model
        self.model.save_weights(checkpoint_filepath)
        self.model.save(checkpoint_filepath)

        return history

    def predict(self, test_data):
        """
        Generates predictions on the provided test data.

        Parameters:
        - test_data (tf.keras.utils.Sequence): Test data generator.

        Returns:
        - y_pred (tf.Tensor): Predicted labels.
        """
        # Generate predictions on test data
        y_pred = self.model.predict(test_data)
        
        # Convert predicted labels to class indices
        y_pred = tf.argmax(y_pred, axis=1)
        return y_pred
    
    def evaluate(self, test_data):
        """
        Evaluates the model on the provided test data.

        Parameters:
        - test_data (tf.keras.utils.Sequence): Test data generator.
        """
        # Evaluate the model on test data
        self.model.evaluate(test_data, return_dict=True)

    def save(self, path):
        """
        Saves the model at the specified path.

        Parameters:
        - path (str): Path to save the model.
        """
        # Save the entire model to the specified path
        self.model.save(path)

    def load(self, path):
        """
        Loads a pre-trained model from the specified path.

        Parameters:
        - path (str): Path to the saved model.
        """
        # Load a pre-trained model from the specified path
        self.model = keras.models.load_model(path)
