'''
*** MLP ***
Binary Classification
with two hidden layers

Author :
Pranath Reddy
2016B5A30572H
'''

import pandas as pd
import math 
import numpy as np
from mat4py import loadmat
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split

def set(y):
    for i in range(len(y)):
        if(y[i]>0.5):
            y[i] = 1.0
        if(y[i]<=0.5):
            y[i] = 0.0
    return y

data = loadmat('./data5.mat')
data = pd.DataFrame(data)
data = np.asarray(data)

data_temp = []
for i in range(len(data)):
    data_temp.append(data[i][0])

data_temp = np.asarray(data_temp)
data = data_temp
y = data[:,-1]
x = data[:,:-1]
x = (x - np.mean(x,axis=0))/np.std(x,axis=0)
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3)
m = x_tr.shape[0]
n = x_tr.shape[1]

for p in range(50,250,25):
        #p - no of neurons in hidden layer

        batch_size = 128
        epochs = 2000

        model = Sequential() # instantiate the model
        model.add(Dense(p, activation='relu', input_shape=(n,))) # first hidden layer
        model.add(Dense(p, activation='relu')) # second hidden layer
        model.add(Dense(1, activation='sigmoid')) # output layer

        sgd = optimizers.SGD(lr=0.01)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=0)
        yp = model.predict(x_ts, batch_size=batch_size)

        yp_temp = []
        for i in range(len(yp)):
            yp_temp.append(yp[i][0])
        yp = yp_temp
        yp = set(yp)

        y_actual = pd.Series(y_ts, name='Actual')
        y_pred = pd.Series(yp, name='Predicted')
        confmat = pd.crosstab(y_actual, y_pred)

        print('Result for hidden neurons : ' + str(p) + ' :')
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







