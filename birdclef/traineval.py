import birdclef.data_preparation as data_prep
import baselinemodel
import lstmmodel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

args = data_prep.args
print("Running arguments: " + args.__str__())

df = pd.read_csv('data/train_metadata.csv')

if args.database_size != 0:
    df = df.head(args.database_size)

label_dict = data_prep.get_label_map(df['scientific_name'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=0)

print('Train size: ', len(train_df))
print('Validation size: ', len(val_df))
print('Test size: ', len(test_df))
print('Total size: ', len(train_df) + len(val_df) + len(test_df))
print('Total size2: ', len(df))

# load data
train_generator = data_prep.BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=args.batch_size, name='train')
val_generator = data_prep.BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=args.batch_size, name='val')
test_generator = data_prep.BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=args.batch_size, name='test')

input_shape = train_generator[0][0].shape[1:]
num_labels = len(label_dict)

# init model
#model = baselinemodel.BaselineModel(args.batch_size, input_shape, num_labels)
model = lstmmodel.LSTMModel(num_labels, input_shape)
# train model

history = model.train(train_generator, val_generator, epochs=2)
print(history.history['accuracy'])
train_accuracy = history.history['accuracy']
epochs = list(range(1, len(train_accuracy) + 1))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='center left')
plt.show()

# test
X_test, y_true = test_generator[0]
y_pred = model.predict(X_test)

# eval metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred,
                            average='weighted')  # Use 'micro', 'macro', or 'weighted' as per your requirement
recall = recall_score(y_true, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted' as per your requirement
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted' as per your requirement

print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1)
