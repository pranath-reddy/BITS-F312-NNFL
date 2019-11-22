'''
*** K-means clustering ***
two clusters

Author :
Pranath Reddy
2016B5A30572H
'''
import pandas as pd
import cmath as math 
import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt

# A function to calculate the mean of an array
def mean(val): 
    if len(val) == 0:
        return 0
    else:
        return sum(val) / len(val)

# A function to return a column of the data at the specified index
def col(array, i):
    return [row[i] for row in array]

# A function to return the max of two values
def higher(x,y):
    if x>y:
        return x
    else:
        return y

# A function to calculate the distance between two points
def dist(x,y):
    sum = 0
    for a in range(4):
        sum = sum + (x[a]-y[a])**2
    return (math.sqrt(sum)).real

# A function to calculate the distances from initialized centroids
def distcen_init(data,cen):
    distc1 = [0 for x in range(len(data))]
    distc2 = [0 for x in range(len(data))]
    for k in range(len(data)):
        distc1[k] = (dist(data[k],data[cen[0]]))
        distc2[k] = (dist(data[k],data[cen[1]]))
    return distc1, distc2

# A function to calculate the distances from centroids
def distcen(data,cen):
    distc1 = [0 for x in range(len(data))]
    distc2 = [0 for x in range(len(data))]
    for k in range(len(data)):
        distc1[k] = (dist(data[k],cen[0]))
        distc2[k] = (dist(data[k],cen[1]))
    return distc1, distc2

def getcol(data,c):
    col = []
    for i in range(len(data)):
        col.append(data[i][c])
    return col
    
# A function to implement min-max normalization
def norm(data):
    ndata = data
    for i in range(4):
        maxval = max(getcol(data,i))
        minval = min(getcol(data,i))
        for j in range(len(data)):
            ndata[j][i] = (data[j][i]-minval)/(maxval-minval)
    return ndata

# import the data
data = pd.read_excel('data2.xlsx',header=None)
# normalize the data
data = np.asarray(data)
data = norm(data)

# initiate the centroids
randindex = [randint(0, len(data)) for b in range(2)] 
dsvc1, dsvc2 = distcen_init(data,randindex)
iters = 50

for j in range(iters):
    # assign cluster indexes
    cval = [1 for x in range(len(data))]
    for l in range(len(data)):
        if dsvc2[l]<dsvc1[l]:
            cval[l] = 2
    # divide into clusters using cluster indexes found above
    clist1 = []
    clist2 = []
    for m in range(len(data)):
        if cval[m] == 1:
            clist1.append(data[m])
        else:
            clist2.append(data[m])
    # update the centroids
    c1 = []
    c2 = []
    for n in range(4):
        c1.append(mean(col(clist1,n)))
        c2.append(mean(col(clist2,n)))
    cen = [c1,c2]
    # update the distances from centroids
    dsvc1, dsvc2 = distcen(data,cen)

index = [0 for x in range(len(data))]
for i in range(len(data)):
    index[i] = i+1

plt.scatter(np.arange(len(data[:,0])),data[:,0],c=cval)
plt.title('Feature 1')
plt.show()
plt.scatter(np.arange(len(data[:,1])),data[:,1],c=cval)
plt.title('Feature 2')
plt.show()
plt.scatter(np.arange(len(data[:,2])),data[:,2],c=cval)
plt.title('Feature 3')
plt.show()
plt.scatter(np.arange(len(data[:,3])),data[:,3],c=cval)
plt.title('Feature 4')
plt.show()

