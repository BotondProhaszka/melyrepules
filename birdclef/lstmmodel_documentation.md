# LSTM Audio Classification Model Documentation

This document provides comprehensive documentation for the LSTM Audio Classification Model script, aimed at training a deep learning model on audio data using Long Short-Term Memory (LSTM) architecture.

## Dependencies

- `tensorflow` (`tf`)
- `keras.layers`
- `keras.models`
- `numpy` (`np`)
- `sklearn.metrics` (`average_precision_score`)

## Functions

### `padded_cmap_tensorflow(y_true, y_pred, padding_factor=5)`

A custom TensorFlow function to calculate the mean average precision score with padding.

Parameters:
- `y_true`: True labels.
- `y_pred`: Predicted labels.
- `padding_factor` (optional): Padding factor for arrays (default is 5).

Returns:
- `result_tensor`: Resulting tensor with the mean average precision score.

### Class: `LSTMModel`

#### Methods:

- `__init__(self, num_labels, input_shape, loss='mean_squared_error', optimizer='adam', metrics='accuracy')`

    Initializes the `LSTMModel` object.

    Parameters:
    - `num_labels`: Number of unique labels for classification.
    - `input_shape`: Shape of the input data.
    - `loss` (optional): Loss function for model training (default is 'mean_squared_error').
    - `optimizer` (optional): Optimizer for model training (default is 'adam').
    - `metrics` (optional): Evaluation metrics for model training (default is 'accuracy').

- `train(self, data, val_data, epochs=100, checkpoint_filepath='saved_model/1', batch_size=30, verbose=1)`

    Trains the model on the provided training data.

    Parameters:
    - `data`: Training data generator.
    - `val_data`: Validation data generator.
    - `epochs` (optional): Number of training epochs (default is 100).
    - `checkpoint_filepath` (optional): Path to save model checkpoints (default is 'saved_model/1').
    - `class_weights` (optional): Class weights for imbalanced data (default is None).
    - `batch_size` (optional): Batch size for training (default is 30).
    - `verbose` (optional): Verbosity mode (default is 1).

    Returns:
    - `history`: Training history.

- `predict(self, test_data)`

    Generates predictions on the provided test data.

    Parameters:
    - `test_data`: Test data generator.

    Returns:
    - `y_pred`: Predicted labels.

- `evaluate(self, test_data)`

    Evaluates the model on the provided test data.

    Parameters:
    - `test_data`: Test data generator.

- `save(self, path)`

    Saves the model at the specified path.

    Parameters:
    - `path`: Path to save the model.

- `load(self, path)`

    Loads a pre-trained model from the specified path.

    Parameters:
    - `path`: Path to the saved model.

## References
To use `class_weights` we used a [tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights) by the original documentation of [TensorFlow](https://www.tensorflow.org/).