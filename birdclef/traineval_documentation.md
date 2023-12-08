# BirdCLEF Audio Classification Script

## Overview

This Python script is designed for audio classification using the BirdCLEF dataset. It encompasses data preparation, model training, and evaluation. The script leverages TensorFlow, pandas, scikit-learn, librosa, and custom modules.

## Functionality

### 1. Data Preparation
   - Loads metadata from 'train_metadata.csv'.
   - Optionally limits the dataset size with the `-s` or `--database_size` command-line argument.
   - Splits the dataset into training, validation, and test sets.
   - Creates data generators for each set.

### 2. Model Architecture
   - Implements three model architectures: `BaselineModel`, `LSTMModel`, and `HuggingModel`.
   - The `LSTMModel` class defines the LSTM model architecture for audio classification.

### 3. Training
   - Initializes the chosen model with the specified parameters.
   - Trains the model using the training set and validates on the validation set.
   - Displays training accuracy over epochs.

### 4. Evaluation
   - Computes various evaluation metrics on the test set:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
      - Custom metric: `padded_cmap` (average precision score with macro averaging and label padding)

### 5. Command-Line Arguments
   - `-s` or `--database_size`: Specifies the dataset size.
   - `-bs` or `--batch_size`: Sets the batch size for training.
   - `-if` or `--input_filename`: Loads a pre-existing file instead of creating a new one.
   - `-of` or `--output_filename`: Sets the output filename for `transform_df`.


## References


