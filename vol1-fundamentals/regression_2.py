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
from tensorflow.keras.datasets import boston_housing

#The data has already been sorted into training and test sets for us
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

#Prepare data: find mean and standard deviation
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data/=std
test_data -= mean
test_data /=std

print(train_data.shape[1])

#Creating the model
def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64,activation = 'relu',input_shape=(train_data.shape[1],)))
    model.add(tf.keras.layers.Dense(64,activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer = "rmsprop",
                loss = "mse",
                metrics = ['mae']
                )
    return model

K = 4
num_val_samples = len(train_data)//K
num_epochs = 100
all_scores = []
all_mae_histories = []
for i in range(K):
    print(f"Processing fold {i}")
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
        train_data[(i+1)*num_val_samples:]],
        axis = 0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
        train_targets[(i+1)*num_val_samples:]],
        axis = 0
    )
    model = build_model()
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data,val_targets),
        epochs=num_epochs,
        batch_size = 1
    )
    all_mae_histories.append(history.history['val_mae'])
    val_mse,val_mae = model.evaluate(val_data,val_targets,verbose = 0)
    all_scores.append(val_mae)

#Finds the average mae within each epoch, takes the average mae from mae's from different partitions
#List comprehension in python: [x[i] for x in all_mae_histories]
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smoothed_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smoothed_mae_history = smoothed_curve(average_mae_history[10:])

# plt.plot(range(1,len(smoothed_mae_history)+1),smoothed_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
print(all_scores)
#Scores without adding validatoin_data when we fit the model
#[2.0716640949249268, 2.6450560092926025, 3.192882537841797, 2.3695685863494873]
print("BUILDING NEW MODEL WITH IMPROVED PARAMETERES...")
model = build_model()
history = model.fit(
        train_data,
        train_targets,
        epochs=28,
        batch_size = 16
    )
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print(f"Test MSE Score: {test_mse_score}")
print(f"Test MAE Score: {test_mae_score}")

