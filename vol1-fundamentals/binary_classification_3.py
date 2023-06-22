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

data = pd.read_csv("/Users/ambroseling/Desktop/TensorFlow/tensorflow-repo/Tensorflow/vol1-Fundamentals/breast-cancer.csv")

#data = pd.get_dummies(data)


X = data.drop(["diagnosis","id"],axis=1)

y = data["diagnosis"].map({'M': 1, 'B': 0})

#Option 2:
#y = data["diagnosis"].replace({'M': 1, 'B': 0})



ct = make_column_transformer(
    (MinMaxScaler(),X.columns.tolist())
)
#Build our train and test sets
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Fit the column transformer to our training data
ct.fit(X_train)
print(f"X train: {X_train}")
#Transform training and test data with normalization (MinMaxScale) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(f"Feature columns: {X_train_normal}")
print(f"Target column: {y_train}")


x_val = X_train_normal[:int(len(X_train_normal)*0.2)]
partial_x_train = X_train_normal[int(len(X_train_normal)*0.2):]

y_val = y_train[:int(len(y_train)*0.2)]
partial_y_train = y_train[int(len(y_train)*0.2):]

print(X_train_normal.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1000,activation = "relu",input_shape=(30,)))
model.add(tf.keras.layers.Dense(1000,activation = "relu"))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(1,activation = "sigmoid"))

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
            metrics = ["accuracy"]
            )

#2. Compile the model
# model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
#                              optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
#                              metrics = ["accuracy"])
#3. Fit the model
# history = model.fit(X,y,epochs=100)

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
model.evaluate(X_test_normal,y_test)
pd.DataFrame(history.history).plot()
plt.show()

#Best accuracy: 97.37%