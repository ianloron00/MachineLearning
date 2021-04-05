# sources: 
# basis pseudocode: https://en.wikipedia.org/wiki/DBSCAN
# another reference (has github, may help): https://medium.com/@darkprogrammerpb/dbscan-clustering-from-scratch-199c0d8e8da1
# graphic: https://towardsdatascience.com/understanding-dbscan-algorithm-and-implementation-from-scratch-c256289479c5
# in this algorithm, it is necessary determine two variables: 
# the ratio distance among points (epsilon value) and the min number of points (minP) in a cluster (C). 
# python3 -m notebook

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math 
import random
import scipy.io as spio
import collections

# find the close points of the current point analysed. (Euclidian referential)
def findNeighbors(data, curr_point, radius):
  points = []
  for neighbor in range(len(data)):
    dist = np.linalg.norm(data.iloc[neighbor]-data.iloc[curr_point])
    if dist <= radius:
      points.append(neighbor)   
  return points

def colorNeighbors(data, curr_points, radius, &labels, &Cluster):
  if len(curr_points) == 0: return
  for q in curr_points:
    if labels[q] != 0: continue
    labels[q] = -1
    Cluster = np.append(q)
    colorNeighbors(data, findNeighbors(data,q,radius), radius, &labels, &Cluster)
  

def recursiveDBSCAN(data, epsilon, minP):
  N = []
  S = []
  C = 0
  point_label = [0]*len(data)
  for p in range(len(data)):
    N = findNeighbors(data, p, epsilon)
    if len(N) >= minP:
      C = C+1
      non_core = np.delete(p)
      point_label[p] = C
      S[C] = np.append(p)
      colorNeighbors(data,N,epsilon,point_label,S[C])
  for o in point_label:
    if point_label[o] = 0:
      S[0] = np.append(o)
  return S,C
 

def dbscan (data, epsilon, minP):
  C = 1
  S = []
  undef = 0
  labeled = -1
  core = -2
  label[0] = np.zeros(shape=len(data), dtype=int)
  label[1] = np.zeros(shape=len(data), dtype=int)
  for p in len(data):
    N = []
    if label[p][0] == core: continue
    elif label[p][0] == undef:
      N = findNeighbors(data,p,epsilon,label)
      for q in N:
        if label[q][0] != undef:
          label[p][1] = label[q][1]
          label[p][0] = labeled
          for s in N:
           label[s][1] = label[p][1]
           label[s][0] = labeled
          break          
      if label[p][0] != undef: continue
      elif (len(N) >= minP):
        C=C+1
        for n in N:
          label[n] = labeled
        label[i] = core
        S = np.append(S,N)
      

    else:

    C=C+1
    else:

  return M

#Function to plot final result
def plotRes(train, data):
  dicColors = {0:'black', 1:'green', 2:'brown', 3:'red', 4:'purple', 5:'orange', 
              6:'yellow', 7:'violet'}

  V = []
  for i in range(len(data)):
     V += [0] * len(data[i])

  for i in range(len(data)):
   for j in range(len(data[i])):
     V[data[i][j]] = i
    
  label_color = [dicColors[c%7] for c in V] # THINK ABOUT
  x_label = 'x'
  y_label = 'y'
  title = 'Plot Graphic Cluster.dat using DBSCAN'
  plt.figure(figsize=(15,15))
  plt.scatter(train.iloc[:,0],train.iloc[:,1],c=label_color,alpha=0.3)
  plt.suptitle(title, fontsize=20)
  plt.suptitle(title, fontsize=20)  
  plt.suptitle(title, fontsize=20)  
  plt.savefig(title + '.jpg')
  plt.show()  

def main():
  #Load Data
  columns = ['coluna 1', 'coluna 2']
  train = pd.read_csv("cluster.dat", decimal=".", sep=' ', names=columns)

  eps = 100
  minpts = 5

  print('Set eps = ' +str(eps)+ ', Minpoints = '+str(minpts))
  pointlabel,cl = recursiveDBSCAN(train,eps,minpts) 
  for i in range (len(pointlabel)):
    print("cluster "+str(i)+": "+str(pointlabel[i]))
  plotRes(train, pointlabel)
  plt.show()
  print('number of cluster found: ' + str(cl-1))
  counter=collections.Counter(pointlabel)
  print(counter)
  outliers  = np.count_nonzero(pointlabel == 0)
  print('numbrer of outliers found: '+str(outliers) +'\n')

main()

# def dbscan (data, epsilon, minP):
  # C = 0
  # N = []
  # S = []
  # M = []
  # label = np.zeros(shape=len(data), dtype=int)
  # for p in range(len(data)):
  #   if (label[p]) != undef : continue
  #   N = findNeighbors(data, p, epsilon) 
  #   if len(N) < minP:
  #     label[p] = noise
  #     continue
  #   C = C+1
  #   label[p] = C
  #   S = np.delete(N, np.where(N == p))
  #   M.append(S)
  #   for q in S:
  #     if label[q] == noise: label[q] = C
  #     if label[q] != undef: continue
  #     label[q] = C
  #     N = findNeighbors(data, q, epsilon)
  #     if len(N) >= minP:
  #       S = np.append(N,S)
  #       M[C-1] = S
  # print("end final loop") # didn't pass
  # return M, C