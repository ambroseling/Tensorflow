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

from sklearn.datasets import make_circles

n_samples = 1000
X,y = make_circles(n_samples,noise = 0.03, random_state=42)

circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)

print(f"X shape: {X.shape} Y shape: {y.shape}")


#set random seed
tf.random.set_seed(42)

#1. Create the model
model = tf.keras.Sequential([
          tf.keras.layers.Dense(4,activation="relu"),   
          tf.keras.layers.Dense(4,activation="relu") , 
            tf.keras.layers.Dense(1,activation="sigmoid")                               

])
#2. Compile the model
model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                             optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
                             metrics = ["accuracy"])
#3. Fit the model
history = model.fit(X,y,epochs=100)
def plot_decision_boundary(model,X,y):
  """
  Plots the decision boundary created by a model predicting on X
  """
  #Define the axis boundaries  of the plot anad create a meshgrid
  x_min,x_max = X[:,0].min() -0.1, X[:,0].max()+0.1
  y_min,y_max = X[:,1].min() -0.1, X[:,1].max()+0.1
  xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))

  #Create X value (we're going to make predictions on these)
  x_in = np.c_[xx.ravel(),yy.ravel()]# stack 2d arrats together

  #Make predictions
  y_pred = model.predict(x_in)

  #Check for multiclass
  if len(y_pred[0])>1:
    print("doing multiclass classification")
    #we have to reshape our prediction to get them ready for plotting
    y_pred = np.argmax(y_pred,axis=1).reshape(xx.shape)
  else: 
    print("doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)

#plots the conouts of the range of values xx and yy
  plt.contourf(xx,yy,y_pred,cmap=plt.cm.RdYlBu,alpha=0.7)
  plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(),xx.max())
  plt.ylim(yy.min(),yy.max())

plot_decision_boundary(model, X, y)
plt.show()

#Plot the loss curves
pd.DataFrame(history.history).plot()
plt.title("Model Binary Classification loss curves")
plt.show()
