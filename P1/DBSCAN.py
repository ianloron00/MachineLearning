# sources: 
# basis pseudocode: https://en.wikipedia.org/wiki/DBSCAN
# another reference (has github, may help): https://medium.com/@darkprogrammerpb/dbscan-clustering-from-scratch-199c0d8e8da1
# graphic: https://towardsdatascience.com/understanding-dbscan-algorithm-and-implementation-from-scratch-c256289479c5

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math 
import random
import scipy.io as spio


# in this algorithm, it is necessary determine two variables: 
# the ratio distance among points (epsilon value) and the min number of points (minP) in a cluster (C). 

noise = -1
undef = 0

# find the close points of the current point analysed. (Euclidian referential)
def findNeighbors(data, curr_point, radius):
  points = []
  for neighbor in range(len(data)):
    if np.linalg.norm(data.iloc[neighbor]-data.iloc[curr_point]) <= radius:
      points.append(neighbor)      
  return points

def dbscan (data, epsilon, minP):
  C = 0
  N = []
  S = []
  label = np.zeros(shape=len(data), dtype=int)
  for p in range(len(data)):
    if (label[p]) != undef : continue
    N = findNeighbors(data, p, epsilon) 
    if len(N) < minP:
      label[p] = noise
      continue

    C = C+1
    label[p] = C

    S = np.delete(N, np.where(N == p))
    for q in range (len(S)):
      if label[q] == noise: label[q] = C
      if label[q] != undef: continue
      label[q] = C
      N = findNeighbors(data, q, epsilon)
      if len(N) >= minP:
        S.append(N)
  return S, C

### graphic - from towardsdatascience ###
#Function to plot final result
def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if (i==0):
            #Plot all noise point as blue
            color='blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')


#Load Data
# raw = spio.loadmat('DBSCAN.mat')
# train = raw['Points']
columns = ['coluna 1', 'coluna 2']
train = pd.read_csv("cluster.dat", decimal=".", sep=' ', names=columns)


#Set EPS and Minpoint
epss = [5,10]
minptss = [5,10]
# Find ALl cluster, outliers in different setting and print resultsw
for eps in epss:
    for minpts in minptss:
        print('Set eps = ' +str(eps)+ ', Minpoints = '+str(minpts))
        pointlabel,cl = dbscan(train,eps,minpts) 
        plotRes(train, pointlabel, cl)
        plt.show()
        print('number of cluster found: ' + str(cl-1))
        counter=collections.Counter(pointlabel)
        print(counter)
        outliers  = pointlabel.count(0)
        print('numbrer of outliers found: '+str(outliers) +'\n')