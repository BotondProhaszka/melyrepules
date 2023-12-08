# Audio Classification Baseline model
This document provides comprehensive documentation for the Audio Classification Neural Network script, aimed at training a deep learning model on audio data, particularly for bird vocalization classification.

## Dependencies
- tensorflow (tf)
- keras.layers
- keras.models
- numpy (np)

## Global Constants
- BATCH_SIZE: Batch size for training.
- input_shape: Shape of the input data.
- num_labels: Number of unique labels for classification.

## Class: BaselineModel
### Methods:
- `__init__(self, BATCH_SIZE, input_shape, num_labels)` Initializes the BaselineModel object.

    - `BATCH_SIZE (int)`: Batch size for training.
    - `input_shape (tuple)`: Input shape of the data.
    - `num_labels (int)`: Number of unique labels for classification.

- `train(self, data, val_data, epochs=100)` Trains the model on the provided training data.

    - `data (tf.keras.utils.Sequence)`: Training data generator.
    - `val_data (tf.keras.utils.Sequence)`: Validation data generator.
    - `epochs (int)`: Number of training epochs.

     Returns: history: Training history.

- `eval(self, test_data)` Evaluates the model on the provided test data.
    - `test_data (tf.keras.utils.Sequence)`: Test data generator.

- `predict(self, test_data)` Generates predictions on the provided test data.

    - `test_data (tf.keras.utils.Sequence)`: Test data generator.

    Returns: `y_pred (tf.Tensor)`: Predicted labels.

## Example Usage
### Instantiate the BaselineModel
```python
from audio_classification import BaselineModel

model = BaselineModel(BATCH_SIZE, input_shape, num_labels)
```

### Train the Model
```python
history = model.train(train_data, val_data, epochs=10)
```
### Evaluate the Model
```python
model.eval(test_data)
```
### Make Predictions
```python
predictions = model.predict(test_data)
```