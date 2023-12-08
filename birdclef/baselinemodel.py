import tensorflow as tf
from keras import layers
from keras.models import Sequential

# Global constants
BATCH_SIZE = 30
input_shape = (1, 160000)
print('Input shape:', input_shape)
num_labels = 243

class BaselineModel():
    """
    Baseline Model for Audio Classification using a Sequential Neural Network.
    """
    def __init__(self, BATCH_SIZE, input_shape, num_labels):
        """
        Initializes the BaselineModel object.

        Parameters:
        - BATCH_SIZE (int): Batch size for training.
        - input_shape (tuple): Input shape of the data.
        - num_labels (int): Number of unique labels for classification.
        """
        self.BATCH_SIZE = BATCH_SIZE
        self.input_shape = input_shape
        self.num_labels = num_labels       

        # Define the Sequential model
        self.model = Sequential([
            layers.Input(shape=self.input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_labels)
        ])

        # Display model summary
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def train(self, data, val_data, epochs=100):
        """
        Trains the model on the provided training data.

        Parameters:
        - data (tf.keras.utils.Sequence): Training data generator.
        - val_data (tf.keras.utils.Sequence): Validation data generator.
        - epochs (int): Number of training epochs.

        Returns:
        - history: Training history.
        """
        history = self.model.fit(
            data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
        return history
    
    def eval(self, test_data):
        """
        Evaluates the model on the provided test data.

        Parameters:
        - test_data (tf.keras.utils.Sequence): Test data generator.
        """
        self.model.evaluate(test_data, return_dict=True)

    def predict(self, test_data):
        """
        Generates predictions on the provided test data.

        Parameters:
        - test_data (tf.keras.utils.Sequence): Test data generator.

        Returns:
        - y_pred (tf.Tensor): Predicted labels.
        """
        y_pred = self.model.predict(test_data)
        print("Predictions")
        print(y_pred)
        y_pred = tf.argmax(y_pred, axis=-1)
        print("Argmaxs")
        print(y_pred)
        return y_pred
