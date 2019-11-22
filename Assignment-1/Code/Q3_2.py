'''
***Multivariate Linear Regression***
With Stochastic Gradient Descent and L2 norm regularization

Author :
Pranath Reddy
2016B5A30572H
'''
import pandas as pd
import math 
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

# A function to return the column at specified index
def getcol(data,c):
    col = []
    for i in range(len(data)):
        col.append(data[i][c])
    return col

# A function to return the updated values of m,c after one iteration of gradient descent
# weight update rule
def wtupdate(m1,m2,c,x1,x2,y,i):
    sumvm1 = 0
    sumvm2 = 0
    sumvc = 0
    lrate = 0.001
    reg = 0.2
    yp = [0 for i in range(len(x1))]
    yp[i] = (m1*x1[i]) + (m2*x2[i]) + c
    sumvm1 = sumvm1 - (y[i]-yp[i])*x1[i]
    sumvm2 = sumvm2 - (y[i]-yp[i])*x2[i]
    sumvc = sumvc - (y[i]-yp[i])
    m1 = m1*(1-lrate*reg) - lrate*sumvm1
    m2 = m2*(1-lrate*reg) - lrate*sumvm2
    c = c*(1-lrate*reg) - lrate*sumvc
    return m1,m2,c

# A function for calculate the cost
def costfn(yp,y,m1,m2,c):
    j = 0
    scale = len(yp)
    reg = 0.2
    for i in range(len(y)):
        j = j + float((yp[i]-y[i]))**2
    j = j + (reg/2)*(m1*m1+m2*m2+c*c)
    return j*0.5*(1/scale)

# A function to return the slope and intercept of y^
def linreg(x1,x2,y):
    m1 = 0
    m2 = 0
    c = 0
    iters = 100
    cost = []
    m1list = []
    m2list = []
    j = 0
    yp = [0 for i in range(len(x1))]
    y_temp = y
    while(j<iters):
        for i in range(len(y)):
            random.shuffle(y_temp)
            m1,m2,c = wtupdate(m1,m2,c,x1,x2,y_temp,i) 
            for i in range(len(x1)):
                yp[i] = (m1*x1[i]) + (m2*x2[i]) + c
        cost.append(costfn(yp,y,m1,m2,c))
        m1list.append(m1)
        m2list.append(m2)
        j = j+1
    return m1,m1list,m2,m2list,c,cost

# A function to implement min-max normalization
def norm(data):
    ndata = data
    for i in range(2):
        maxval = max(getcol(data,i))
        minval = min(getcol(data,i))
        for j in range(len(data)):
            ndata[j][i] = (data[j][i]-minval)/((maxval-minval)+0.05)
    return ndata

# import the data
data = pd.read_excel('data.xlsx',header=None)
# normalize the data
data = np.asarray(data)
data = norm(data)

# split into dependent and independent variables
x1 = data[:,0]
x2 = data[:,1]
y = data[:,-1]

#run the linear regression
m1,m1list,m2,m2list,c,cost = linreg(x1,x2,y)

plt.plot(cost)
plt.title("cost vs iterations")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(m1list, m2list, cost, 'blue')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')
ax.set_title('cost vs weights')
fig.show()
plt.show()

  



