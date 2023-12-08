import data_preparation as data_prep
import baselinemodel
import lstmmodel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
import numpy as np

def padded_cmap_numpy(y_true, y_pred, padding_factor=5):
    # Pad true and predicted labels to account for potential false positives at the beginning
    y_true = np.pad(y_true, (padding_factor, 0), constant_values=1)
    y_pred = np.pad(y_pred, (padding_factor,  0), constant_values=1)
    # Calculate average precision score using macro averaging
    return average_precision_score(
        y_true.astype(int),
        y_pred,
        average="macro",
    )

# Load command-line arguments
args = data_prep.args
print("Running arguments: " + args.__str__())

# Load metadata DataFrame
df = pd.read_csv('data/train_metadata.csv')

# Limit the database size if specified
if args.database_size != 0:
    df = df.head(args.database_size)

# Create a dictionary to map scientific names to numerical labels
label_dict = data_prep.get_label_map(df['scientific_name'])

# Split the dataset into training, validation, and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=0)

print('Train size: ', len(train_df))
print('Validation size: ', len(val_df))
print('Test size: ', len(test_df))
print('Total size: ', len(train_df) + len(val_df) + len(test_df))
print('Total size2: ', len(df))

# Create data generators for training, validation, and testing
train_generator = data_prep.BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=args.batch_size, name='train')
val_generator = data_prep.BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=args.batch_size, name='val')
test_generator = data_prep.BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=args.batch_size, name='test')

# Get input shape and number of labels for model initialization
input_shape = train_generator[0][0].shape[1:]
num_labels = len(label_dict)

# Initialize the LSTM model
model = lstmmodel.LSTMModel(num_labels, input_shape)

# Train the model and get training history
history = model.train(train_generator, val_generator, epochs=2)
print(history.history['accuracy'])

# Plot training and validation accuracy over epochs
train_accuracy = history.history['accuracy']
epochs = list(range(1, len(train_accuracy) + 1))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='center left')
plt.show()

# Test the model on the test set
X_test, y_true = test_generator[0]
y_pred = model.predict(X_test)

# Evaluate model performance using various metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
padded_cmap = padded_cmap_numpy(y_true, y_pred, 10)

# Print evaluation metrics
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1)
print('padded_cmap: ', padded_cmap)
