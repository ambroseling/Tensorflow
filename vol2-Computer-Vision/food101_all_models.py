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

#======================MODEL 1: Feature Extraction transfer learning model with 10% of data augmentation======================
# 83% accuracy
train_dir_10_percent = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/train"
test_dir_10_percent = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_10_percent/test"
train_dir_100_percent = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_all_data/train"
test_dir_100_percent = "/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol2-Computer-Vision/10_food_classes_all_data/test"

IMG_size = (224,224)
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_10_percent,
                                                                            label_mode = "categorical",
                                                                            image_size = IMG_size)
test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(test_dir_10_percent,
                                                                label_mode = "categorical",
                                                                image_size = IMG_size)
#Create model 2 with data augmentation built in
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

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

#Create the inputs and outputs 
inputs = tf.keras.Input(shape = input_shape,name= "input_layer")
x = data_augmentation(inputs) #augment our training images (augmentation doesnt occur on test data)
x = base_model(x,training=False)#pass augmented images to base model but keeps it in inference mode
x = layers.GlobalAveragePooling2D(name="globale_average_pooling_2D")(x)
outputs = layers.Dense(10,activation = "softmax",name = "output_layer")(x)

model_1 = tf.keras.Model(inputs,outputs)

#Compile
model_1.compile(loss="categorical_crossentropy",
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

#Fitting the model
history_model_1 = model_1.fit(train_data_10_percent,
                        epochs =  initial_epochs ,
                        steps_per_epoch = len(train_data_10_percent),
                        validation_data = test_data_10_percent,
                        validation_steps = int(0.25*len(test_data_10_percent)),
                        callbacks = [create_tensorboard_callback(dir_name = "transfer_learning",experiment_name="10_percent_data_aug"),checkpoint_callback] )
print("MODEL 1 EVALUATION:")
model_1.evaluate(test_data_10_percent)

#======================MODEL 2: Fine-tuning an existing model on 10% of data======================
# 85% accuracy
#To begin fine tuning , lets start by setting the last 10 layers of our base_model.trainable to True
base_model.trainable = True

#Freeze all layers except for the last 10 
for layer in base_model.layers[:-10]:
  layer.trainable = False
  #print(layer.name)

#Recompile the model (we have to recompile our models every time we make changes)
model_1.compile(loss = "categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                #in fine-tuning you typically want to lower your learning rate by 10x
                metrics = ["accuracy"])
#Fine tune for another 5 epochs
fine_tune_epochs = initial_epochs +5
#Refit the model (same as model_2 except with more trainable layers)
history_model_2 = model_1.fit(train_data_10_percent,
                                               epochs = fine_tune_epochs,
                                               steps_per_epoch = len(train_data_10_percent),
                                               validation_data = test_data_10_percent,
                                               validation_steps = int(0.25*len(test_data_10_percent)), #will validate on 25% of the test data
                                               initial_epoch = history_model_1.epoch[-1],#start training from previous last epoch
                                               callbacks = [create_tensorboard_callback(dir_name = "transfer_learning",experiment_name = "10_percent_fine_tune_last_10")]
                                                )
print("MODEL 2 EVALUATION:")
model_1.evaluate(test_data_10_percent)
#======================MODEL 3: Fine-tuning an existing model on 10% of data======================
# 87% accuracy
train_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(train_dir_100_percent,
                                                                                 label_mode = "categorical",
                                                                                 image_size = IMG_size) 

test_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(test_dir_100_percent,
                                                                                 label_mode = "categorical",
                                                                                 image_size = IMG_size) 

model_1.load_weights(checkpointpath)
#Compile 
model_1.compile(loss = "categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                metrics = ["accuracy"])

#Continue to train and fine-tune the model to our model (100% of training data)
fine_tune_epochs = initial_epochs +5
history_model_3 = model_1.fit(train_data_10_classes_full,
                                           epochs = fine_tune_epochs,
                                           validation_data = test_data_10_classes_full,
                                           validation_steps = int(0.25*len(test_data_10_classes_full)),
                                           initial_epoch = history_model_2.epoch[-1],
                                           callbacks = [create_tensorboard_callback(dir_name = "transfer_learning",experiment_name = "full_10_classes_fine_tune_last_10")])
print("MODEL 3 EVALUATION:")
model_1.evaluate(test_data_10_classes_full)
