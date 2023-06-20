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
from helper_functions import plot_loss_curves,create_tensorboard_callback,compare_historys,make_confusion_matrix


#=========================MODEL 1: Big dog with transfer learning on 10% data=========================

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

# Steps we going to take:
# Create a modelcheckpoint callback
# Creata data augmentation layer
# Build a headless (no top layer) funcitonal efficientNetB0 back-boned model
# Compile our model
# Feature extract for 5 full passes (5 epochs on the train dataset and validata on 15% of the test data to save epoch time)


#Create checkpoint callback
checkpoint_path = "ten_percent_model_checkpoints_weights_big_dog"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weigjhts_only = True,
                                                         monitor = "val_accuracy",
                                                         save_best_only=True)

#Set up data augmentation
data_augmentation = Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
],name = "data_augmentation_layer")

#Setup base model and freeze its layers (this will extract features)
base_model  = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

#Set up model architecture with trainable top layer
inputs = layers.Input(shape = (224,224,3),name = "input_layer")
x = data_augmentation(inputs)#augment images only
x = base_model(x,training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pooling_layer")(x)
#len(train_data_all_10_percent.class_name)=10
outputs = layers.Dense(len(train_data_10_percent.class_names),activation="softmax",name="output_layer")(x)
model = tf.keras.Model(inputs,outputs)


#compile the model
model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["accuracy"])

#Fit the model
history_model_1 = model.fit(train_data_10_percent,
                                           epochs = 5,
                                           validation_data= test_data_10_percent,
                                           validation_steps = int(0.15*len(test_data_10_percent)),
                                           callbacks = [checkpoint_callback])

moel_1_results = model.evaluate(test_data_10_percent)
plot_loss_curves(history_model_1)
plt.show()

#=========================MODEL 2: Big dog with transfer learning and fine tuning on 10% data=========================

#unfreeze all of the layers in the base model
base_model.trainable = True

#Refreeze every layer except the last 5
for layer in base_model.layers[:-5]:
  layer.trainable = False
#layers closer to the output layer get unfrozen/ fine-tuned first

#Recompile the model with lower learning rate (typically best practice to do so during fine tuning)
model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ["accuracy"])
#fine-tune for 5 more epochs
fine_tune_epochs = 10 #model has already bene trained for 5 (feature extraction),this is the total number of epochs

history_model_2 = model.fit(train_data_10_percent,
                                                       epochs = fine_tune_epochs,
                                                       validation_data = test_data_10_percent,
                                                       validation_steps = int(0.15*len(test_data_10_percent)),
                                                     initial_epoch = history_model_1.epoch[-1])
model_2_results = model.evaluate(test_data_10_percent)
compare_historys(original_history = history_model_1 ,new_history=history_model_2,initial_epochs = 5)
plt.show()
#========================= Loaded big dog and make predictions =========================

model.save("food101_big_dog_model.h5")
loaded_model = tf.keras.models.load_model("food101_big_dog_model.h5")
loaded_model_results = loaded_model.evaluate(test_data_10_percent)
#Make predictions with model
pred_probs = model.predict(test_data_10_percent,verbose=1) #set verbosity to see how long is left

print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like: {pred_probs[0]}")
print(f"The class with the. highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")
print(f"Predicted class for sample 0 is: {test_data_10_percent.class_names[pred_probs[0].argmax()]}")

#get the pred classess of each label
#an array of the highest probabilites of all 25250 data sets

pred_classes = pred_probs.argmax(axis=1)
#To get our test labels we need to unravel our test_data batchdataset
y_labels = []
for images , labels in test_data_10_percent.unbatch():
  y_labels.append(labels.numpy().argmax())


make_confusion_matrix(y_true=y_labels,
                      y_pred=pred_classes,
                      classes = class_names,
                      figsize=(100,100),
                      text_size = 20)

#Lets try sci-kit learns accuracy score function and see the results
from sklearn.metrics import accuracy_score,classification_report
sklearn_accuracy = accuracy_score(y_true=y_labels,
                                  y_pred=pred_classes)
class_names = test_data_10_percent.class_names
classification_report_dict = classification_report(y_labels,pred_classes,output_dict = True)
#Lets plot all of our classes F1 scores
class_f1_scores = {}
#Loop through classification report dictionary items
for k,v in classification_report_dict.items():
  print(k)
  if k == "accuracy": #stop once we get to accuracy key (the last key because we dont need this, we just want f1 scores)
    break
  else:
#Add class names and f1 scores to new dictionary
    class_f1_scores [class_names[int(k)]] = v["f1-score"]



plt.show()