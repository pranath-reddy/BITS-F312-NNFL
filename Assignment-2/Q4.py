'''
*** ELM ***
Binary Classification

Author :
Pranath Reddy
2016B5A30572H
'''

import pandas as pd
import math 
import numpy as np
from mat4py import loadmat
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

for p in range (250,1001,50):

        randommat = np.random.randn(x_tr.shape[1]+1,p)
        H = np.append(np.ones((x_tr.shape[0],1)), x_tr, axis=1)
        H = np.dot(H,randommat)
        H = np.tanh(H)
        H = np.matrix(H)

        y_tr = np.matrix(y_tr)
        W = np.dot(H.I,y_tr.T)

        H_ts = np.append(np.ones((x_ts.shape[0],1)), x_ts, axis=1)
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











