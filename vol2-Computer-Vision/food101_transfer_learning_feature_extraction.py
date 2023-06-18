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
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

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

#Build a data augmentation layer
data_augmentation = Sequential([
   #preprocessing.Rescaling(1./255),
   preprocessing.RandomFlip("horizontal"),
   preprocessing.RandomHeight(0.2),
   preprocessing.RandomZoom(0.2),
   preprocessing.RandomWidth(0.2),
   preprocessing.RandomRotation(0.2)                             

], name = "data_augmentation")

#Set up the input shape to our model 
input_shape = (224,224,3)
#Create a frozen base model (also called the back bone)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable=False

#Freeze all layers except for the last 10 
for layer in base_model.layers[:-10]:
  layer.trainable = False

#Create the inputs and outputs 
inputs = tf.keras.Input(shape = input_shape,name= "input_layer")
x = data_augmentation(inputs) #augment our training images (augmentation doesnt occur on test data)
x = base_model(x,training=False)#pass augmented images to base model but keeps it in inference mode
x = layers.GlobalAveragePooling2D(name="globale_average_pooling_2D")(x)
outputs = layers.Dense(10,activation = "softmax",name = "output_layer")(x)

model = tf.keras.Model(inputs,outputs)

#Compile
model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

#Set checkpoing path
checkpointpath = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"

#Create a modelcheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointpath,
                                                         save_weights_only = True,
                                                         save_best_only = False,
                                                         save_freq = "epoch",#save every epoch
                                                         verbose = 1)

#Fit the model saving checkpoints every epoch
initial_epochs = 5
fine_tune_additional_epoch = 5
#Fitting the model
history= model.fit(train_data_10_percent,
                        epochs =  initial_epochs+fine_tune_additional_epoch,
                        steps_per_epoch = len(train_data_10_percent),
                        validation_data = test_data,
                        validation_steps = int(0.25*len(test_data)),
                        callbacks = [create_tensorboard_callback(dir_name = "transfer_learning",experiment_name="10_percent_data_aug"),checkpoint_callback] )
model.evaluate(test_data)

