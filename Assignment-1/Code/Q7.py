'''
***Logistic Regression***
With Gradient Descent

Author :
Pranath Reddy
2016B5A30572H
'''

import math 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# A function to return the column at specified index
def getcol(data,c):
    col = []
    for i in range(len(data)):
        col.append(data[i][c])
    return col

def set(y):
    for i in range(len(y)):
        if(y[i]>0.5):
            y[i] = 1
        if(y[i]<0.5):
            y[i] = 0
    return y

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# A function to return the updated values of m,c after one iteration of gradient descent
def wtupdate(m1,m2,m3,m4,c,x,y):
    sumvm1 = 0
    sumvm2 = 0
    sumvm3 = 0
    sumvm4 = 0
    sumvc = 0
    yp = [0 for i in range(len(x))]
    for i in range(len(x)):
        yp[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + c
        yp[i] = sigmoid(yp[i])
        sumvm1 = sumvm1 - (y[i]-yp[i])*x[i,0]
        sumvm2 = sumvm2 - (y[i]-yp[i])*x[i,1]
        sumvm3 = sumvm3 - (y[i]-yp[i])*x[i,2]
        sumvm4 = sumvm4 - (y[i]-yp[i])*x[i,3]
        sumvc = sumvc - (y[i]-yp[i])
    m1 = m1 - 0.05*sumvm1
    m2 = m2 - 0.05*sumvm2
    m3 = m3 - 0.05*sumvm3
    m4 = m4 - 0.05*sumvm4
    c = c - 0.05*sumvc
    return m1,m2,m3,m4,c

# A function to return the slope and intercept of y^
def linreg(x,y):
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    c = 0
    iters = 1000
    i = 0
    while(i<iters):
        m1,m2,m3,m4,c = wtupdate(m1,m2,m3,m4,c,x,y)
        i = i+1
    return m1,m2,m3,m4,c

# A function to implement min-max normalization
def norm(data):
    ndata = data
    for i in range(5):
        maxval = max(getcol(data,i))
        minval = min(getcol(data,i))
        for j in range(len(data)):
            ndata[j][i] = (data[j][i]-minval)/((maxval-minval)+0.05)
    return ndata

# import the data
data = pd.read_excel('data3.xlsx',header=None)
# normalize the data
data = np.asarray(data)
data = norm(data)

# split into dependent and independent variables
x = data[:,:-1]
y = data[:,-1]

# split into testing and training sets
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.4)
m1,m2,m3,m4, c = linreg(x_tr,y_tr)
x = x_ts
yp = [0 for i in range(len(x))]
for i in range(len(x)):
    yp[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + c
    yp[i] = sigmoid(yp[i])
y_ts = set(y_ts)
yp = set(yp)
y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)
print(confmat)
confmat = np.asarray(confmat)
tp = confmat[1][1]
tn = confmat[0][0]
fp = confmat[0][1]
fn = confmat[1][0]

Acc = (tp+tn)/(tp+tn+fp+fn)
SE = tp/(tp+fn)
SP = tn/(tn+fp)

print('Accuracy : ' + str(Acc))
print('sensitivity : ' + str(SE))
print('specificity : ' + str(SP))



   


