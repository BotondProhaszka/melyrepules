# Bird Voice AI Trainer Documentation

This document provides comprehensive documentation for the Bird Voice AI Trainer script, designed for training a deep learning model on bird vocalization data.

## Dependencies

- `random`
- `pandas` (`pd`)
- `numpy` (`np`)
- `os`
- `librosa`
- `glob`
- `math`
- `tensorflow` (`tf`)
- `argparse`
- `matplotlib.pyplot` (`plt`)

## Command-line Arguments

The script accepts the following command-line arguments:

- `-s` or `--database_size`: How big will be the database. All data will be loaded if its value is 0.
- `-bs` or `--batch_size`: Batch size for training.
- `-if` or `--input_filename`: If not empty, the script will load a file instead of creating a new one with `transform_df`. Do not use a file extension (It's csv).
- `-of` or `--output_filename`: Output filename of `transform_df`. Only needed if `input_filename` is empty. Do not use a file extension (It's csv).

## Functions and Classes

### `argbuilder()`

Function to build and parse command-line arguments.

### `get_label_map(df_column)`

Function to create a label map based on unique values in a DataFrame column.

### `BirdCLEF_DataGenerator(tf.keras.utils.Sequence)`

Class for generating data batches for training a BirdCLEF model. Inherits from `tf.keras.utils.Sequence`.

#### Methods:

- `__init__(self, df, label_dict, root_folder, name, batch_size=10, shuffle=True)`: Initializes the data generator.
- `__len__(self)`: Returns the number of batches in the sequence.
- `__getitem__(self, index)`: Generates one batch of data.
- `get_wawe_len(self, filepath)`: Gets the length of the audio file in seconds.
- `transform_df(self, df)`: Transforms the input DataFrame based on command-line arguments.
- `on_epoch_end(self)`: Called at the end of each epoch.
- `open_wawe(self, filepath)`: Opens the audio file and returns the waveform.
- `frame_audio(self, audio_array, window_size_s=5.0, hop_size_s=5.0, sample_rate=32000)`: Helper function for framing audio for inference.
- `resample(self, waveform, desired_sample_rate, original_sample_rate)`: Resamples the waveform to the desired sample rate.
- `ensure_sample_rate(self, waveform, original_sample_rate, desired_sample_rate=32000)`: Ensures the sample rate of the waveform.
- `data_analysis(self, df)`: Performs data analysis on the input DataFrame. It calculates the duration of each audio file, generates a histogram of the audio lengths, and saves it as 'histogram.png'. It also counts the occurrences of scientific names and saves the counts as a DataFrame in 'scientific_names_counts.csv'. The analysis provides insights into the distribution of audio lengths and the frequency of scientific names in the dataset.

### `extract_features(audio_path, sr=32000, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512)`

Function to extract certain features from an audio file.

#### Parameters:
- `audio_path`: Path to the audio file.
- `sr`: Sampling rate (default is 32000).
- `n_mfcc`: Number of MFCC coefficients (default is 13).
- `n_mels`: Number of mel filterbanks (default is 128).
- `n_fft`: Number of FFT points (default is 2048).
- `hop_length`: Hop length for feature extraction (default is 512).

#### Returns:
- `features`: Extracted features for Linear and Logistic Regression.

## Example Usage

### Instantiate the BirdCLEF_DataGenerator

```python
import tensorflow as tf
from data_preparation import *

args = argbuilder()
df = pd.read_csv('train_metadata.csv')
label_dict = get_label_map(df['scientific_name'])
train_generator = BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=args.batch_size, name='train')
```
### Data analysis


```python
data_gen.data_analysis(df)
```