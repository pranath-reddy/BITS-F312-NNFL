'''
*** Deep layer stacked autoencoder based ELM ***
Binary Classification

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
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
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

# Pretraining Models
# Layer 1
input1 = Input(shape = (n, ))
encoder1 = Dense(120, activation = 'sigmoid')(input1)
decoder1 = Dense(n, activation = 'sigmoid')(encoder1)

autoencoder1 = Model(input = input1, output = decoder1)
encoder_layer1 = Model(input = input1, output = encoder1)

# Layer 2
input2 = Input(shape = (120,))
encoder2 = Dense(100, activation = 'sigmoid')(input2)
decoder2 = Dense(120, activation = 'sigmoid')(encoder2)

autoencoder2 = Model(input = input2, output = decoder2)

sgd = optimizers.SGD(lr=0.1)

autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd)
autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd)

encoder_layer1.compile(loss='binary_crossentropy', optimizer = sgd)

autoencoder1.fit(x_tr, x_tr, epochs= 2000, batch_size = 128, verbose=0)
layer2_input = encoder_layer1.predict(x_tr)
autoencoder2.fit(layer2_input, layer2_input, epochs= 2000, batch_size = 128, verbose=0)

encoder1_sa = Dense(120, activation = 'sigmoid')(input1)
encoder2_sa = Dense(100, activation = 'sigmoid')(encoder1_sa)

stack_autoencoder = Model(input = input1, output = encoder2_sa)

stack_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd)

stack_autoencoder.layers[1].set_weights(autoencoder1.layers[1].get_weights()) 
stack_autoencoder.layers[2].set_weights(autoencoder2.layers[1].get_weights()) 

elm_train_input = stack_autoencoder.predict(x_tr, batch_size=128)
elm_test_input = stack_autoencoder.predict(x_ts, batch_size=128)


for p in range (250,1001,50):
        #p - no of neurons in hidden layer

        randommat = np.random.randn(elm_train_input.shape[1]+1,p)
        H = np.append(np.ones((elm_train_input.shape[0],1)), elm_train_input, axis=1)
        H = np.dot(H,randommat)
        H = np.tanh(H)
        H = np.matrix(H)

        y_tr = np.matrix(y_tr)
        W = np.dot(H.I,y_tr.T)

        H_ts = np.append(np.ones((elm_test_input.shape[0],1)), elm_test_input, axis=1)
        H_ts = np.dot(H_ts,randommat)
        H_ts = np.tanh(H_ts)
        H_ts = np.matrix(H_ts)
        y_pred = np.dot(H_ts,W)

        y_pred_temp = []
        y_pred = np.asarray(y_pred)
        for i in range(y_pred.shape[0]):
            y_pred_temp.append(y_pred[i][0])
        yp = set(y_pred_temp)

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


















































