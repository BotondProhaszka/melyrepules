import data_preparation as data_prep
import lstmmodel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
import numpy as np
import tensorflow as tf
import hugging

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

# Create a dictionary to map numerical labels to class weights
class_weights = pd.read_csv('../birdclef/scientific_names_counts.csv')
print(label_dict)
class_weights['scientific_name'] = class_weights['scientific_name'].map(label_dict)
class_weights = class_weights.set_index('scientific_name')
class_weights = class_weights.to_dict()['counts']

#Scale weights
total = sum(class_weights.values())
num_classes = len(class_weights)
class_weights = {key: (1 / value) * (total / num_classes) for key, value in class_weights.items()}

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

model = None

# Train the model and get training history
if args.train:
    if args.hug:
        model = hugging.HuggingModel(input_shape, label_dict)
        history = model.train(train_generator, val_generator)
    else:
        model = lstmmodel.LSTMModel(num_labels, input_shape)
        history = model.train(train_generator, val_generator, class_weights=class_weights, epochs=args.epoch,
                              checkpoint_filepath="./saved_model/" + args.model_filename)
    print(history.history['accuracy'])
else:
    # Initialize the LSTM model
    loaded_model = tf.keras.models.load_model("./saved_model/" + args.model_filename)
    loaded_model.summary()


# Plot training and validation accuracy over epochs
if args.train:
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
