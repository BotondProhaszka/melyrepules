import random
import pandas as pd
import numpy as np
import os
import librosa

from sklearn.model_selection import train_test_split

import tensorflow as tf
import glob




def get_label_map(df_column):
    labels = set(df_column)
    label_map = {label: i for i, label in enumerate(labels)}
    print(df_column.value_counts())
    return label_map



class BirdCLEF_DataGenerator():
  def __init__(self, df, label_dict, root_folder, batch_size=10, shuffle=True):
    
    self.batch_size = batch_size

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
        sounds.append(wawe)
        labels.append(self.label_dict[row['scientific_name']])
        
    return sounds, labels
  
  def on_epoch_end(self):
      self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)



  def open_wawe(self, filepath):
    audio, sample_rate = librosa.load(self.rootfolder + '/train_audio/' +filepath)
    sample_rate, wav_data = self.ensure_sample_rate(audio, sample_rate)
    wav_data = self.frame_audio(wav_data)

  def frame_audio(self,
      audio_array: np.ndarray,
      window_size_s: float = 5.0,
      hop_size_s: float = 5.0,
      sample_rate = 32000,
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


train_generator = BirdCLEF_DataGenerator(train_df, label_dict,'data/', batch_size=BATCH_SIZE)
val_generator = BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=BATCH_SIZE)
test_generator = BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=BATCH_SIZE)


a = train_generator[0]
b = train_generator[1]

print('Dict: ', label_dict)

print('Wrong sample num: ', train_generator.wrong_sample_num)

