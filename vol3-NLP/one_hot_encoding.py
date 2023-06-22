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
from  tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten, Dense,Activation
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

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


samples = ['The cat sat on the mat.','The dog ate my homework.']

#=============================One hot encoding with words=============================

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word]=len(token_index)+1
max_length = 10
results = np.zeros(shape =( len(samples),
                    max_length,
                    max(token_index.values())+1))
for i,sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index]=1

#=============================One hot encoding with characters=============================
characters = string.printable
token_index = dict(zip(range(1,len(characters)+1),characters))

max_length = 50
results = np.zeros(
    ((len(samples)),
    max_length,
    max(token_index.keys())+1
))
for i,sample in enumerate(samples):
    for j,character in enumerate(sample):
        index = token_index.get(character)
        results[i,j,index]=1

#=============================One hot encoding with keras=============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
print("Sequences:")
print(sequences)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')
print("Results:")
print(one_hot_results.shape)
word_index = tokenizer.word_index

