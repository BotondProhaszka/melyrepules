import gradio as gr
import librosa
import data_preparation as data_prep
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd

def make_batches(data, batch_size):
    batches = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def recogniseBird(file):
    
    #result = model.predict(file)
    if ".ogg" not in file:
        return "töltsön fel .ogg kiterjesztésű file-t"
    audio, sample_rate = librosa.load(file)
    sample_rate, wav_data = data_generator.ensure_sample_rate(audio, sample_rate)
    wav_data = data_generator.frame_audio(wav_data)
    batches = make_batches(wav_data, 10)
    prediction_counts = {}
    for batch in batches:
        batch = np.expand_dims(batch,1)
        predicted_labels = model.predict(batch)
        predicted_labels = np.argmax(predicted_labels, axis=1)
        print("predicted_label: ", predicted_labels)
        for label in predicted_labels:
            if label not in prediction_counts:
                    prediction_counts[label] = 0
            prediction_counts[label] += 1
    most_common_label = max(prediction_counts, key=prediction_counts.get)
    return most_common_label

data_generator = data_prep.BirdCLEF_DataGenerator(pd.DataFrame(), {},"","generator")
demo = gr.Interface(fn=recogniseBird, inputs="file", outputs="text")
model_path = "../saved_model/2"
model = load_model(model_path)

if __name__ == "__main__":
    demo.launch() 