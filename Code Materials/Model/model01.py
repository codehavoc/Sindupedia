import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

#Keras
import keras
from keras import models
from keras import layers
from keras.constraints import maxnorm
# from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json

csvFile_extracted_features = 'dataset_features_balanced.csv'
genres = ['classics','dance','disco','hiphop','pop','rnb','soul','snj']
numofgenres = len(genres)

dataset1 = pd.read_csv(csvFile_extracted_features)
dataset1.head()

# Dropping unneccesary columns
columns_to_drop = []
# columns_to_drop = ['filename','chroma_stft','rmse','zero_crossing_rate','mfcc5','mfcc7','mfcc8','mfcc9','mfcc14','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']
columns_to_drop.append('filename')
dataset1 = dataset1.drop(columns_to_drop,axis=1)
dataset1.head()

print("extracted data without file names : ",dataset1)

# Encoding genres into integers
genre_list_of_extracted_features = dataset1.iloc[:, -1]
encoder = LabelEncoder()
y1 = encoder.fit_transform(genre_list_of_extracted_features)

# normalizing
scaler = StandardScaler()
X1 = scaler.fit_transform(np.array(dataset1.iloc[:, :-1], dtype = float))

# Data train test splitting
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.4, random_state=42, stratify=y1)
X_test1, X_validate1, y_test1, y_validate1 = train_test_split(X_test1, y_test1, test_size=0.5, random_state=42, stratify=y_test1)

np.random.seed(23456)
tf.random.set_seed(123)

# creating a model
model_ef = models.Sequential()

model_ef.add(layers.Dense(256, activation='relu', input_shape=(X_train1.shape[1],)))

model_ef.add(layers.Dropout(0.2)) 

model_ef.add(layers.Dense(128, activation='relu', kernel_constraint=maxnorm(3)))

model_ef.add(layers.Dropout(0.2))

model_ef.add(layers.Dense(64, activation='relu', kernel_constraint=maxnorm(3)))

model_ef.add(layers.Dropout(0.2))

model_ef.add(layers.Dense(16, activation='relu', kernel_constraint=maxnorm(3)))

model_ef.add(layers.Dense(8, activation='softmax'))

model_ef.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
history = model_ef.fit(X_train1,
                    y_train1,
                    epochs=40,
                    validation_data=(X_validate1, y_validate1),
                    batch_size=16)

# calculate accuracy
test_loss, test_acc = model_ef.evaluate(X_test1,y_test1)
print('test_acc: ',test_acc)
print('test_loss: ',test_loss)

# ==============================================================================================
# serialize model to JSON
# model_json = model_ef.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to HDF5
# model_ef.save_weights("model.h5")
# print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# loaded_model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
              
# test_loss, test_acc = loaded_model.evaluate(X_test1,y_test1)
# print('test_acc: ',test_acc)
# print('test_loss: ',test_loss)
# ==============================================================================================
# predictions
predictions = model_ef.predict(X_test1)

print(f"\nthere are prediction {len(predictions)} results\n")

print(y_test1[0:10])

for i in range(0,10):
  print(genres[np.argmax(predictions[i])])

# Confusion Matrix
labels_dict = {
    0: 'classics',
    1: 'dance',
    2: 'disco',
    3: 'hiphop',
    4: 'pop',
    5: 'rnb',
    6: 'soul',
    7: 'snj'
}

