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
from keras.datasets import imdb
#RUN WITH python3 binary_classification.py !!!! 

# data = pd.read_csv("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol1-Fundamentals/breast-cancer.csv")

# X = data.drop(["diagnosis"],axis=1)
# y = data["diagnosis"]

# print(X)

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

#num_words = 10000 means you only keep the top 10000 most frequently occuring words in the training data, discarding rare words
#max word index is 10000
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decoded_review)

#Vectorize the data
def vectorize_sequences (sequence,dimension=10000):
    results = np.zeros((len(sequence),dimension))
    for i,sequence in enumerate(sequence):
        results[i,sequence]=1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#Vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16,activation = 'relu',input_shape = (10000,)))
model.add(tf.keras.layers.Dense(16,activation = 'relu'))
model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',
            optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
            metrics = ['accuracy']
            )
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size=512,
    validation_data=(x_val,y_val)
)

pd.DataFrame(history.history).plot()
plt.show()