import sys
import os
import tensorflow.keras
import pathlib
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
import matplotlib.image as mpimg
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten, Dense,Activation

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

#Walk thorugh 10 percent data directory and list number of files
for dirpath,dirnames,filenames in os.walk("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent"):
  print(f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


#Set up data inputs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper_functions import plot_loss_curves,create_tensorboard_callback



train_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/train"
test_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/test"

IMG_size = (224,224)
BATCH_size = 32
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size = IMG_size,
                                                                            label_mode = "categorical",
                                                                            batch_size = BATCH_size)
test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                            image_size = IMG_size,
                                                                            label_mode = "categorical",
                                                                            batch_size = BATCH_size)

#Create a base model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top = False)

#Freeze the base model (so the underlying pre-trained patterns aren't updated)
base_model.trainable = False

#Create inputs into our model
inputs = tf.keras.layers.Input(shape=(224,224,3),name="input_layer")

#If using ResNet50V2 you will need to normalize inputs
x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

#Pass the inputs to the base_model
x = base_model(inputs)
print(f"Shape after passing inputs through base model: {x.shape}")

#Average pool the outputs of the base model (aggregrate all the most important informaation,reduce number of computation)
x = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling_layer")(x)
#this layer is a representation of our input data that the trained model has learnerd due to its own patterns
print(f"Shape after Global Average Pooling 2D: {x.shape}")

#Create the output activation layer
outputs = tf.keras.layers.Dense(10,activation = "softmax",name = "output")(x)

#Combine the inputs with the outputs into a model
model = tf.keras.Model(inputs,outputs)

#Compile the model
model.compile(loss = "categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

#Fit the model and save its history
history = model.fit(train_data_10_percent,
                                 epochs = 5,
                                 steps_per_epoch = len(train_data_10_percent),
                                 validation_data = test_data,
                                 validation_steps = int(0.25*len(test_data)),
                                 callbacks = [create_tensorboard_callback(dir_name = "transfer_learning",
                                                                          experiment_name = "10_percent_feature_extraction")])
                         