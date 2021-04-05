import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math 
import random
import scipy.io as spio
import collections

columns = ['coluna 1', 'coluna 2']
train = pd.read_csv("cluster.dat", decimal=".", sep=' ', names=columns)
eps = 0.01
minpts = 5

def normalize(data):
  normData = data.copy()
  for i in range(data.shape[1]):
    normData.iloc[:,i] /= np.linalg.norm(data.iloc[:,i])
  return normData

# find the close points of the current point analysed. (Euclidian referential)
def findNeighbors(data, curr_point, radius):
  points = []
  for neighbor in range(len(data)):
    dist = np.linalg.norm(data.iloc[neighbor]-data.iloc[curr_point])
    if dist <= radius:
      points.append(neighbor)   
  return points

# Iteratively Color Neighbors
def ICN(data, curr_points, radius, labels, Cluster):
  for p in curr_points:
    if labels[p] == 0:
      Cluster = np.append(Cluster,p)
      labels[p] = -1
    N = findNeighbors(data,p,radius)
    for q in N:
      if labels[q] != 0: continue
      labels[q] = -1
      Cluster = np.append(Cluster,q)
      curr_points.append(q)  
  return labels, Cluster

def DBSCAN(data, epsilon, minP):
  N = []
  S = [[]]
  C = 0
  point_label = np.zeros(len(data),dtype=int) 
  for p in range(len(data)):
    if point_label[p] != 0: continue
    N = findNeighbors(data, p, epsilon)
    if len(N) >= minP:
      C = C+1
      point_label[p] = -1
      S.append([p]) 
      point_label, S[C] = ICN(data,N,epsilon,point_label,S[C])
      # print('Cluster '+str(p)+' (size '+str(len(S[C]))+'): '+str(S[C]))
  for o in range(len(point_label)):
    if point_label[o] == 0:
      S[0] = np.append(S[0],o)
  return S

#Function to plot final result
def plotRes(train, data, main_title):
  dicColors = {0:'black', 1:'green', 2:'blue', 3:'red', 4:'purple', 5:'orange', 
              6:'yellow', 7:'violet', 8:'brown'}
  V = [0] * len(train)
  for i in range(len(data)):
   for j in range(len(data[i])):
    V[data[i][j]] = i
    
  label_color = [dicColors[c%8] for c in V] 
  x_label = 'x axis'
  y_label = 'y axis'
  title = main_title 
  plt.figure(figsize=(15,15))
  plt.scatter(train.iloc[:,0],train.iloc[:,1],c=label_color,alpha=0.3)
  plt.suptitle(title, fontsize=20)
  plt.suptitle(title, fontsize=20)  
  plt.suptitle(title, fontsize=20)  
  plt.savefig(title + '.png')
  plt.show()

def callClusters(train_data,eps,minpts):
  print('Set epsilon (normalized radius) = ' +str(eps)+ ', Min Points = '+str(minpts))
  return DBSCAN(train_data,eps,minpts)

def callPlot(train_data,pointlabel, main_title):
  cl = len(pointlabel)
  # for i in range (len(pointlabel)):
  #   print("cluster "+str(i)+": "+str(pointlabel[i]))
  plotRes(train_data, pointlabel, main_title) 
  plt.show()
  print('number of cluster found: ' + str(cl-1))
  counter=collections.Counter(pointlabel)
  print(counter)
  outliers  = np.count_nonzero(pointlabel == 0)
  print('number of outliers found: '+str(outliers) +'\n')

def main():
  normTrain = normalize(train)
  pt_label = callClusters(normTrain,eps,minpts)
  callPlot(train,pt_label,
           'Clusters division applying method DBSCAN (epsilon 0.1) - data from cluster.dat')

main()