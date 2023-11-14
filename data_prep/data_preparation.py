import random
import pandas as pd
import numpy as np
import os
import librosa
import glob
import math

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl



def get_label_map(df_column):
    labels = set(df_column)
    label_map = {label: i for i, label in enumerate(labels)}
    print(df_column.value_counts())
    return label_map

 # :) -> not really

class BirdCLEF_DataGenerator():
    def __init__(self, df, label_dict, root_folder, batch_size=10, shuffle=True):

        self.train_df = None
        self.batch_size = batch_size

        df = self.transform_df(df)
        self.df = df
        self.n = len(self.df)
        self.label_dict = label_dict
        self.rootfolder = root_folder
        self.wrong_sample_num = 0
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, index):
        # A kívánt indexű sorok kiválasztása
        batch_df = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        # A hangfájlok betöltése
        sounds = []
        labels = []
        for index, row in batch_df.iterrows():
            wawe = self.open_wawe(row['filename'])
            for i in range(len(wawe)):
                sounds.append(wawe[i])
                labels.append(self.label_dict[row['scientific_name']])
        
        sounds = tf.convert_to_tensor(sounds)
        print("Get_item - Sound shape: ", sounds.shape)
        labels = tf.convert_to_tensor(labels)
        print("Get_item - Labels shape: ", labels.shape)
        return sounds, labels
    
    def get_wawe_len(self, filepath):
        try:
                audio, sample_rate = librosa.load('data/' + '/train_audio/' + filepath)
                wav_len = math.ceil(librosa.get_duration(y=audio, sr=sample_rate)/5)
                return wav_len 

        except Exception as e:
                print(f"Error processing {filepath}: {e}")
                return None

    def transform_df(self, df):
        new_df = df.copy()
        new_df['audio_length'] = new_df['filename'].apply(lambda x: self.get_wawe_len(filepath=x) )

        new_df_expanded = new_df.loc[new_df.index.repeat(new_df['audio_length'])].reset_index(drop=True)
        new_df_expanded['audio_part'] = new_df_expanded.groupby('filename').cumcount()
        return new_df_expanded

    def on_epoch_end(self):
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)

    def open_wawe(self, filepath):
        audio, sample_rate = librosa.load(self.rootfolder + '/train_audio/' + filepath)
        sample_rate, wav_data = self.ensure_sample_rate(audio, sample_rate)
        wav_data = self.frame_audio(wav_data)
        return wav_data

    def frame_audio(self,
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

    def resample(self, waveform, desired_sample_rate, original_sample_rate):
        """Resample waveform if required without tfio."""
        # Kiszámítja a resampled jel hosszát.
        resampled_signal_length = int(len(waveform) * desired_sample_rate / original_sample_rate)

        # Létrehoz egy resampled jel tárolására szolgáló tömböt.
        resampled_signal = np.zeros(resampled_signal_length)

        # A resampled jel minden pontjához kiszámítja a megfelelő értéket.
        for i in range(resampled_signal_length):
            # Kiszámítja a megfelelő indexet az eredeti jelben.
            original_signal_index = int(i * original_sample_rate / desired_sample_rate)

            # Megkapja az eredeti jel értékét a megfelelő indexen.
            original_signal_value = waveform[original_signal_index]

            # Beállítja a resampled jel értékét a megfelelő indexen.
            resampled_signal[i] = original_signal_value

        # Visszaadja a resampled jel tömböt.
        return resampled_signal

    def ensure_sample_rate(self, waveform, original_sample_rate,
                           desired_sample_rate=32000):
        """Resample waveform if required without tfio."""
        if original_sample_rate != desired_sample_rate:
            self.wrong_sample_num = self.wrong_sample_num + 1
            waveform = self.resample(waveform, desired_sample_rate, original_sample_rate)
        return desired_sample_rate, waveform


def extract_features(audio_path, sr=32000, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512):
    """Extracting certain features for Linear- and LogisticRegression"""
    # Hangfájl betöltése
    y, sr = librosa.load(audio_path, sr=sr)

    # Tulajdonságok kinyerése
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Az összes jellemző átlagának és szórásának kinyerése
    features = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                          np.mean(mel, axis=1), np.std(mel, axis=1),
                          np.mean(zcr), np.std(zcr),
                          np.mean(chroma, axis=1), np.std(chroma, axis=1)))

    return features


def linear_regression(dfr, target_column='scientific_name'):
    """LinearRegression from dfr dataframe"""
    # target column enkódolása lineáris regresszióhoz
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dfr[target_column])

    # Jellemzők kinyerése minden hangfájlhoz
    X = [extract_features('data/train_audio/' + row['filename']) for index, row in dfr.iterrows()]

    # Adatok felosztása tanító és teszthalmazra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Lineáris regresszió fitting
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predikció
    predictions = model.predict(X_test)

    # Modell értékelése
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)


def logistic_regression(dfr, target_column='scientific_name'):
    """LogisticRegression from dfr dataframe"""
    y = dfr[target_column]

    # Jellemzők kinyerése minden hangfájlhoz
    X = [extract_features('data/train_audio/' + row['filename']) for index, row in dfr.iterrows()]

    # Adatok felosztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Logisztikus regresszió fitting
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    # Predikció
    predictions = model.predict(X_test)

    # Modell értékelése
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)


BATCH_SIZE = 10

df = pd.read_csv('data/train_metadata.csv')

label_dict = get_label_map(df['scientific_name'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=0)

print('Train size: ', len(train_df))
print('Validation size: ', len(val_df))
print('Test size: ', len(test_df))
print('Total size: ', len(train_df) + len(val_df) + len(test_df))
print('Total size2: ', len(df))

train_generator = BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=BATCH_SIZE)
val_generator = BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=BATCH_SIZE)
test_generator = BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=BATCH_SIZE)

a = train_generator[0]
#b = train_generator[1]

#print('Dict: ', label_dict)

#print('Wrong sample num: ', train_generator.wrong_sample_num)



print('A SHAPE:')

#print(a)# [sounds][sound of bird in 5 second snipets][sound snipets]

'''def train_model(config=None):

    train_generator = BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=BATCH_SIZE)
    val_generator = BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=BATCH_SIZE)
    test_generator = BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=BATCH_SIZE)

    model = Model((160000), len(label_dict))

    checkpoint_callback = pl.callbacks.ModelCheckpoint()
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", patience=3, verbose=False, mode="max")

    trainer = pl.Trainer(max_epochs=10,
                        callbacks=[checkpoint_callback, early_stop_callback]
    )
    # Train the model
    trainer.fit(model, train_generator)

    # Evaluate the model
    trainer.test(dataloaders= test_generator)'''