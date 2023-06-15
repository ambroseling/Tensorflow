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
from tensorflow.keras.datasets import fashion_mnist

#The data has already been sorted into training and test sets for us
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()

#Create a small list so we can index onto our training labels so they're human readable
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","AnkleBoot"]

# For our multiclass classification model, we can use a similar architecture to our binary classifiers, however, we're going to have to tweak a few things

# Input shape = 28x28 (the shape of one thing)
# Output shape = 10 (one per class of clothing)
# Loss Function tf.keras.losses.CategoricalCrossentropy()
# Output layer activation = Softmax (not sigmoid)
# If your labels are one-hot encoded, use CategoricalCrossentropy()

# If your labels are integer form use SparseCategoricalentropy()

x_val = train_data[:1000]/255.
partial_x_train = train_data[1000:]/255.

y_val = to_categorical(train_labels[:1000])
partial_y_train = to_categorical(train_labels[1000:])

x_test = test_data/255.
y_test = to_categorical(test_labels)


tf.random.set_seed(42)

#Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.compile(
    loss = tf.keras.losses.CategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr=0.003),
    metrics=["accuracy"]
)

#Create the learning rate callback
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3*10**(epoch/20))


history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=128,
    validation_data=(
        x_val,y_val
    ),
  #  callbacks=[lr_scheduler]
)



model.evaluate(x_test,y_test)
pd.DataFrame(history.history).plot()

plt.show()

lrs = 1e-3 * (10**(tf.range(25)/20))
plt.plot(lrs,history.history["loss"])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")
plt.show()
#Best accuracy so far is 88%