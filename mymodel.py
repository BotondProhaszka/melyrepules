import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.Sequential

BATCH_SIZE = 10
input_shape = (1, 160000)
print('Input shape:', input_shape)
num_labels = 67

class MyModel():
    def __init__(self, BATCH_SIZE, input_shape, num_labels):
        
        self.BATCH_SIZE = BATCH_SIZE
        self.input_shape = input_shape
        self.num_labels = num_labels       

        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_labels),
        ])

        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def train(self, data, val_data, epochs = 100):
        history = self.model.fit(
            data=data,
            validation_data=val_data,
            epochs=epochs,
            callvacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
        return history
    
    def eval(self, test_data):
        self.model.evaluate(test_data, return_dict=True)

    def predict(self, test_data):
        y_pred = self.model.predict(test_data)
        return y_pred