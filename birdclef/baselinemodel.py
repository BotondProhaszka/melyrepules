import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential

BATCH_SIZE = 30
input_shape = (1, 160000)
print('Input shape:', input_shape)
num_labels = 243


class BaselineModel():
    def __init__(self, BATCH_SIZE, input_shape, num_labels):
        
        self.BATCH_SIZE = BATCH_SIZE
        self.input_shape = input_shape
        self.num_labels = num_labels       

        self.model = Sequential([
            layers.Input(shape=self.input_shape),
            # Downsample the input.
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_labels)
        ])

        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def train(self, data, val_data, epochs = 100):
        history = self.model.fit(
            data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
        return history
    
    def eval(self, test_data):
        self.model.evaluate(test_data, return_dict=True)

    def predict(self, test_data):
        y_pred = self.model.predict(test_data)
        y_pred = tf.argmax(y_pred, axis=1)
        return y_pred
