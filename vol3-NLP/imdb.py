import sys
import os
import pathlib
import platform
import random
import string 

import tensorflow as tf
import tensorflow.keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten, Dense,Activation,Embedding
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import preprocessing

import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
sys.path.insert(1, '/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow')
import helper_functions


max_features = 10000
maxlen = 20

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = max_features)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen = maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)

model = Sequential()
model.add(Embedding(10000,8,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['acc'])
model.summary()
history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size = 32,
                    validation_split=0.2)




#===============================================================================
imdb_dir = "/Users/ambroseling/Desktop/Tensorflow/tensorflow-repo/Tensorflow/vol3-NLP/aclImdb"
train_dir = os.path.join(imdb_dir,"train")

labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            f = open(os.path.join(dir_name,fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(f"Sequences shape: {type(sequences[0][0])}")
word_index = tokenizer.word_index
print(f"Word_index shape: {type(word_index)}")
#print('Found %s unique tokens.'%len(word_index))
data = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=maxlen)
labels = np.asarray(labels)

#[23,3,4531,435,231,54,31,5,...] => one sentence


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]



