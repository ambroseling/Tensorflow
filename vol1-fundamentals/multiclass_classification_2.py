import sys
import tensorflow.keras
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.utils import to_categorical

import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

#Lists are not tensors, which is why train_data.shape doesnt show the inner dimensions of each element 
def get_newswire(data,index):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in data[index]])
    return decoded_newswire

#Vectorize the data
def vectorize_sequences (sequence,dimension=10000):
    results = np.zeros((len(sequence),dimension))
    for i,sequence in enumerate(sequence):
        results[i,sequence]=1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels,dimension = 46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


#Create the model, use higher number of hidden units
model = models.Sequential()
model.add(tf.keras.layers.Dense(64,activation="relu",input_shape=(10000,)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(46,activation="softmax"))

model.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy']
                )
#Pick out validation set
x_val = x_train[:1000] #first 1000 samples
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs = 9,#tried 20, starts to overfit at 9 epochs
    batch_size = 512,
    validation_data=(
        x_val,y_val
    )
)

model.evaluate(x_test,one_hot_test_labels)
predictions = model.predict(x_test)
print(f"Decoded text: {get_newswire(test_data,11)} ")
print(f"Prediction: {tf.keras.datasets.reuters.get_label_names()[np.argmax(predictions[11])]}")
print(f"Actual label: {tf.keras.datasets.reuters.get_label_names()[np.argmax(one_hot_test_labels[11])]}")
pd.DataFrame(history.history).plot()
plt.show()