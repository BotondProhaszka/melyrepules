import random
import pandas as pd
import numpy as np
import os
import librosa

import tensorflow as tf
import glob


class MyDataGenerator():
  def __init__(self, dffilepath, batch_size=10, shuffle=True, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    self.rootfolder = dffilepath
    self.df = pd.read_csv(dffilepath + 'train_metadata.csv')
    self.batch_size = batch_size
    self.n = len(self.df)
    
    self.wrong_sample_num = 0

    self.label_dict = self.get_label_map(self.df['scientific_name'])

    if shuffle:
      self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    # Az adatok szétválasztása a train, val és test részekre
    self.train_df, self.val_df, self.test_df = self.split_data(train_ratio, val_ratio, test_ratio)

    print('Train size: ', len(self.train_df))
    print('Validation size: ', len(self.val_df))
    print('Test size: ', len(self.test_df))
    print('Total size: ', len(self.train_df) + len(self.val_df) + len(self.test_df))
    print('Total size2: ', len(self.df))

  def split_data(self, train_ratio, val_ratio, test_ratio):
      # Az egyes részek mérete
      train_size = int(self.n * train_ratio)
      val_size = int(self.n * val_ratio)
      test_size = self.n - train_size - val_size

      # A df DataFrame véletlenszerű megkeverése
      self.df = self.df.sample(frac=1).reset_index(drop=True)

      # A train, val és test részek kivágása
      train_df = self.df[:train_size]
      val_df = self.df[train_size:train_size + val_size]
      test_df = self.df[train_size + val_size:]

      return train_df, val_df, test_df




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


  def get_label_map(self, df_column):
    labels = set(df_column)
    label_map = {label: i for i, label in enumerate(labels)}
    print(df_column.value_counts())
    return label_map


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

datagenerator = MyDataGenerator('/app/data/')

a = datagenerator[0]
b = datagenerator[1]

print('Dict: ', datagenerator.label_dict)

print('Wrong sample num: ', datagenerator.wrong_sample_num)