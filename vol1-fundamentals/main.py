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

data = pd.read_csv("/content/sample_data/income_data.csv")

X = data.drop(["happiness","Unnamed: 0"],axis=1)

y = data["happiness"]


ct = make_column_transformer(
    (MinMaxScaler(),["income"])
)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# X_validation = X_train_normal[-100:]
# y_validation = y_train_normal[-100:]

tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(1000))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1))


model.compile(loss = tf.keras.losses.mae,
            optimizer = tf.keras.optimizers.Adam(),
            metrics=["mae"])
model_history = model.fit(
    X_train_normal,y_train,epochs=100
)
model.evaluate(X_test_normal,y_test)

test = [0.3,0.4,0.5]
prediction = model.predict(test)

# Print the predicted value
print(prediction)

#sets the size of the plot
plt.figure(figsize = (10,7))
#Plot training data in blue
plt.scatter(X_train,y_train, c="b", label = "Training data")
plt.scatter(X_test,y_test, c="g", label = "Testing data")
#Show a legend
plt.legend()
pd.DataFrame(model_history.history).plot()

