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
IMG_SHAPE = (224,224)
BATCH_SIZE = 32

train_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/train"
test_dir = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/test"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Trainging images")
train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size = IMG_SHAPE,
                                                          batch_size = BATCH_SIZE,
                                                          class_mode = "categorical")

print("Testing images")
test_data_10_percent = train_datagen.flow_from_directory(test_dir,
                                                          target_size = IMG_SHAPE,
                                                          batch_size = BATCH_SIZE,
                                                          class_mode = "categorical")

#Lets compare the following 2 models

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

#Lets make a crea_model() function to create a model from url

def create_model(model_url,num_classes = 10):
  """
  Takes a Tensorflow hub url and creates a Keras Sequential model with it.

  Args: 
  model_url : A tensorflow hub feature extration URL
  num_classes (int): Number of output neurons in the output layer,should be equal to the number of target classes

  Returns:
  An uncompiled Keras sequential model with model_url as feature extractor
  Layer and Dense output layer with num_classes

  """
  #Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           name = "Feature_Extraction_Layer",
                                           input_shape = IMG_SHAPE+(3,))#Freeze the already learned patterns
  #Create our own model
  model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(num_classes,activation="softmax",name = "output_layer")                         
  ])
  return model


#Transfer learning model 1: RESENET
#Create resnet model
resnet_model = create_model(resnet_url,num_classes=10)

#Compile the resnet model
resnet_model.compile(loss = "categorical_crossentropy",
                     optimizer = tf.keras.optimizers.Adam(),
                     metrics = ["accuracy"])

resnet_history=resnet_model.fit(train_data_10_percent,
                 epochs=5,
                 steps_per_epoch=len(train_data_10_percent),
                 validation_data = test_data_10_percent,
                 validation_steps = len(test_data_10_percent),
                 callbacks = [create_tensorboard_callback(dir_name="tensorflow_hub",
                                                           experiment_name="resnet50V2")]
                 )
plot_loss_curves(resnet_history)


#Transfer learning model 1: EfficientNetB0
#Create EfficientNetB0 faeture extractor model
efficientnet_model = create_model(model_url = efficientnet_url,
                               num_classes=train_data_10_percent.num_classes)

#Compile efficientnet model
efficientnet_model.compile(loss = "categorical_crossentropy",
                     optimizer = tf.keras.optimizers.Adam(),
                     metrics = ["accuracy"])

#Fit EfficientNet model to 10% of training data
efficientnet_history = efficientnet_model.fit(
                 train_data_10_percent,
                 epochs=5,
                 steps_per_epoch=len(train_data_10_percent),
                 validation_data = test_data_10_percent,
                 validation_steps = len(test_data_10_percent),
                 callbacks = [create_tensorboard_callback(dir_name="tensorflow_hub",
                                                           experiment_name="efficient_net_b0")])
plot_loss_curves(efficientnet_history)

# tensorboard dev upload --logdir ./tensorflow_hub/ \
#   --name "EfficientNetb0 V.S. ResNet50V2" \
#   --description "Comparing two different TF Hub feature extraction model architecture using 10% of the training data" \
#   --one_shot

#Check out what tensorboard experiments you have
#!tensorboard dev list