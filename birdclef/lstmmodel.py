import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential


class LSTMModel():
    
    def init (self, num_labels, input_shape, loss = 'category_crossentropy', optimizer = 'adam', metrics = 'accuracy'):

        self.num_labels = num_labels
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer


        self.model = Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(128, activation='relu'),
            layers.Dense(self.num_labels)
        ])

        self.model.summary()
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
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

