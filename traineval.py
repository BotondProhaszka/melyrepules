import data_prep.data_preparation as data_prep
import mymodel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



BATCH_SIZE = 10

df = pd.read_csv('data/train_metadata.csv')

df = df.head(100)
label_dict = data_prep.get_label_map(df['scientific_name'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=0)

print('Train size: ', len(train_df))
print('Validation size: ', len(val_df))
print('Test size: ', len(test_df))
print('Total size: ', len(train_df) + len(val_df) + len(test_df))
print('Total size2: ', len(df))

train_generator = data_prep.BirdCLEF_DataGenerator(train_df, label_dict, 'data/', batch_size=BATCH_SIZE)
val_generator = data_prep.BirdCLEF_DataGenerator(val_df, label_dict, 'data/', batch_size=BATCH_SIZE)
test_generator = data_prep.BirdCLEF_DataGenerator(test_df, label_dict, 'data/', batch_size=BATCH_SIZE)

input_shape = train_generator[0][0].shape[1:]
num_labels = len(label_dict)

mymodel = mymodel.MyModel(BATCH_SIZE, input_shape, num_labels)

history = mymodel.train(train_generator, val_generator, epochs=100)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='middle left')
plt.show()

#test
y_pred = mymodel.predict(test_generator)

