import gradio as gr
import librosa
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import defaultdict

def make_batches(data, batch_size):
    batches = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def frame_audio(
                audio_array: np.ndarray,
                window_size_s: float = 5.0,
                hop_size_s: float = 5.0,
                sample_rate=32000,
                ) -> np.ndarray:

    """Helper function for framing audio for inference."""
    """ using tf.signal """
    if window_size_s is None or window_size_s < 0:
        return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)
    return framed_audio

def ensure_sample_rate(waveform, original_sample_rate,
                           desired_sample_rate=32000):
        """
        Ensure the sample rate of the waveform.
        """
        if original_sample_rate != desired_sample_rate:
            waveform = resample(waveform, desired_sample_rate, original_sample_rate)
        return desired_sample_rate, waveform

def resample(waveform, desired_sample_rate, original_sample_rate):
        """
        Resample waveform if required without tfio.
        """
        resampled_signal_length = int(len(waveform) * desired_sample_rate / original_sample_rate)
        resampled_signal = np.zeros(resampled_signal_length)

        for i in range(resampled_signal_length):
            original_signal_index = int(i * original_sample_rate / desired_sample_rate)
            original_signal_value = waveform[original_signal_index]
            resampled_signal[i] = original_signal_value

        return resampled_signal

def get_label_map(df_column):
    labels = set(df_column)
    label_map = {label: i for i, label in enumerate(labels)}
    print(df_column.value_counts())
    return label_map

def recogniseBird(file):
    
    if ".ogg" not in file:
        return "töltsön fel .ogg kiterjesztésű file-t"
    audio, sample_rate = librosa.load(file)
    sample_rate, wav_data = ensure_sample_rate(audio, sample_rate)
    wav_data = frame_audio(wav_data)
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

    return num_dict[most_common_label]

df = pd.read_csv('data/train_metadata.csv')
label_dict = get_label_map(df['scientific_name'])
num_dict = defaultdict(int)
for key, value in label_dict.items():
    num_dict[value] = key
demo = gr.Interface(fn=recogniseBird, inputs="file", outputs="text")
model_path = "../saved_model/2"
model = load_model(model_path)

if __name__ == "__main__":
    demo.launch() 