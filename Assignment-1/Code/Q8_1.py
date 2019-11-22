'''
***Logistic Regression***
one vs all multiclass

Author :
Pranath Reddy
2016B5A30572H
'''
import pandas as pd
import math 
import numpy as np
from sklearn.model_selection import train_test_split

# A function to return the column at specified index
def getcol(data,c):
    col = []
    for i in range(len(data)):
        col.append(data[i][c])
    return col

def set(y):
    for i in range(len(y)):
        if(y[i]>=0.5):
            y[i] = 1
        if(y[i]<0.5):
            y[i] = 0
    return y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# A function to return the updated values of m,c after one iteration of gradient descent
def wtupdate(m1,m2,m3,m4,m5,m6,m7,c,x,y):
    sumvm1 = 0
    sumvm2 = 0
    sumvm3 = 0
    sumvm4 = 0
    sumvm5 = 0
    sumvm6 = 0
    sumvm7 = 0
    sumvc = 0
    yp = [0 for i in range(len(x))]
    for i in range(len(x)):
        yp[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + (m5*x[i,4]) + (m6*x[i,5]) + (m7*x[i,6]) + c
        yp[i] = sigmoid(yp[i])
        sumvm1 = sumvm1 - (y[i]-yp[i])*x[i,0]
        sumvm2 = sumvm2 - (y[i]-yp[i])*x[i,1]
        sumvm3 = sumvm3 - (y[i]-yp[i])*x[i,2]
        sumvm4 = sumvm4 - (y[i]-yp[i])*x[i,3]
        sumvm5 = sumvm5 - (y[i]-yp[i])*x[i,4]
        sumvm6 = sumvm6 - (y[i]-yp[i])*x[i,5]
        sumvm7 = sumvm7 - (y[i]-yp[i])*x[i,6]
        sumvc = sumvc - (y[i]-yp[i])
    m1 = m1 - 0.1*sumvm1
    m2 = m2 - 0.1*sumvm2
    m3 = m3 - 0.1*sumvm3
    m4 = m4 - 0.1*sumvm4
    m5 = m5 - 0.1*sumvm5
    m6 = m6 - 0.1*sumvm6
    m7 = m7 - 0.1*sumvm7
    c = c - 0.1*sumvc
    return m1,m2,m3,m4,m5,m6,m7,c

# A function to return the slope and intercept of y^
def linreg(x,y):
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    m5 = 0
    m6 = 0
    m7 = 0
    c = 0
    iters = 2000
    i = 0
    while(i<iters):
        m1,m2,m3,m4,m5,m6,m7,c = wtupdate(m1,m2,m3,m4,m5,m6,m7,c,x,y)
        i = i+1
    return m1,m2,m3,m4,m5,m6,m7,c

# A function to implement min-max normalization
def norm(data):
    ndata = data
    for i in range(7):
        maxval = max(getcol(data,i))
        minval = min(getcol(data,i))
        for j in range(len(data)):
            ndata[j][i] = (data[j][i]-minval)/((maxval-minval)+0.05)
    return ndata


# import the data
data = pd.read_excel('data4.xlsx',header=None)
data = np.asarray(data)
y = data[:,-1]
data = norm(data)
x = data[:,:-1]


# split into testing and training sets
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.4)

y1_tr = [1 for i in range(len(y_tr))]
y2_tr = [1 for i in range(len(y_tr))]
y3_tr = [1 for i in range(len(y_tr))]
for i in range(len(y_tr)):
    if(y_tr[i] != 1):
        y1_tr[i] = 0
    if(y_tr[i] != 2):
        y2_tr[i] = 0
    if(y_tr[i] != 3):
        y3_tr[i] = 0

x = x_ts

m1,m2,m3,m4,m5,m6,m7,c = linreg(x_tr,y1_tr)
yp1 = [0 for i in range(len(x))]
for i in range(len(x)):
    yp1[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + (m5*x[i,4]) + (m6*x[i,5]) + (m7*x[i,6]) + c
    yp1[i] = sigmoid(yp1[i])
yp1 = set(yp1)

m1,m2,m3,m4,m5,m6,m7,c = linreg(x_tr,y2_tr)
yp2 = [0 for i in range(len(x))]
for i in range(len(x)):
    yp2[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + (m5*x[i,4]) + (m6*x[i,5]) + (m7*x[i,6]) + c
    yp2[i] = sigmoid(yp2[i])
yp2 = set(yp2)

m1,m2,m3,m4,m5,m6,m7,c = linreg(x_tr,y3_tr)
yp3 = [0 for i in range(len(x))]
for i in range(len(x)):
    yp3[i] = (m1*x[i,0]) + (m2*x[i,1]) + (m3*x[i,2]) + (m4*x[i,3]) + (m5*x[i,4]) + (m6*x[i,5]) + (m7*x[i,6]) + c
    yp3[i] = sigmoid(yp3[i])
yp3 = set(yp3)

cval = [0 for i in range(len(y_ts))]
for i in range(len(y_ts)):
    if (yp1[i] == 1):
        cval[i] = 1.0
    if (yp2[i] == 1):
        cval[i] = 2.0
    if (yp3[i] == 1):
        cval[i] = 3.0


for i in range(len(cval)):
    if (cval[i] == 0):
        cval[i] = 'None'
y_actual = pd.Series(y_ts, name='Actual')
y_pred = pd.Series(cval, name='Predicted')
confmat = pd.crosstab(y_actual, y_pred)
print(confmat)

confmat = np.asarray(confmat)
Acc = (confmat[0][0] + confmat[1][1] + confmat[2][2])/sum(sum(confmat))
Acc1 = confmat[0][0]/sum(confmat[0])
Acc2 = confmat[1][1]/sum(confmat[1])
Acc3 = confmat[2][2]/sum(confmat[2])
print('Overall Accuracy : ' + str(Acc))
print('Accuracy of class 1 : ' + str(Acc1))
print('Accuracy of class 2 : ' + str(Acc2))
print('Accuracy of class 3 : ' + str(Acc3))







   


