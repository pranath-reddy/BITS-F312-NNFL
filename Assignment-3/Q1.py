'''
*** CNN ***
Binary Classification

Author :
Pranath Reddy
2016B5A30572H
'''

from mat4py import loadmat
import numpy as np 
import pandas as pd
from keras import Sequential
from keras import optimizers
from keras.layers import Dense,Conv1D,AveragePooling1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

x = loadmat('data_for_cnn.mat')
x = pd.DataFrame(x)
x = np.asarray(x)

x_temp = []
for i in range(len(x)):
    x_temp.append(x[i][0])
x_temp = np.asarray(x_temp)
x = x_temp
x = preprocessing.normalize(x)
x = x.reshape(x.shape[0],x.shape[1],1)

y = loadmat('class_label.mat')
y = pd.DataFrame(y)
y = np.asarray(y)

y_temp = []
for i in range(len(y)):
    y_temp.append(y[i][0][0])
y_temp = np.asarray(y_temp)
y = y_temp

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3)
x_tr = x_tr.reshape(700,1000,1)
x_ts = x_ts.reshape(300,1000,1)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Input-Convolution Layer-Pooling layer- Convolution Layer-Pooling layer -FC1-FC2-FC3-Output
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(5), input_shape=(1000,1), strides=2, padding='valid', activation='relu'))
model.add(AveragePooling1D(pool_size=2,strides=2,padding='same'))
model.add(Conv1D(filters=32, kernel_size=(5), strides=2, padding='valid', activation='relu'))
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.summary()
model.fit(x_tr,y_tr, epochs=epochs)

# testing the model
yp = model.predict_classes(x_ts)

y_temp3 = []
for i in range(len(yp)):
    y_temp3.append(yp[i][0])
y_temp3 = np.asarray(y_temp3)
yp = y_temp3

y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)

print(confmat)
confmat = np.asarray(confmat)
tp = confmat[1][1]
tn = confmat[0][0]
fp = confmat[0][1]
fn = confmat[1][0]
Acc = float(tp+tn)/float(tp+tn+fp+fn)
SE = float(tp)/float(tp+fn)
SP = float(tn)/float(tn+fp)
print('Accuracy : ' + str(Acc))
print('sensitivity : ' + str(SE))
print('specificity : ' + str(SP))

