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


(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#Data preprocessing (normalize btw 0 and 1)
train_images= train_images.reshape((60000,28,28,1))
train_images= train_images.astype('float32')/255
test_images_copy = test_images
test_images= test_images.reshape((10000,28,28,1))
test_images= test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
network.add(tf.keras.layers.MaxPooling2D(2,2))
network.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
network.add(tf.keras.layers.MaxPooling2D(2,2))
network.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
network.add(tf.keras.layers.Flatten())
network.add(tf.keras.layers.Dense(64,activation="relu"))
network.add(tf.keras.layers.Dense(10,activation="sigmoid"))



network.compile(optimizer = "rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])
network.fit(train_images,train_labels,epochs=5,batch_size = 64)
network.evaluate(test_images,test_labels)

prediction = network.predict(test_images)
print(f"Prediction: {prediction[25]}")
digit = test_images_copy[25]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()