import tensorflow as tf

# Modell betöltése (mentés mappáját kell megadni)
loaded_model = tf.keras.models.load_model("./saved_model/1")
loaded_model.summary()

# A betöltött modell használata
# predictions = loaded_model.predict(input_data)
