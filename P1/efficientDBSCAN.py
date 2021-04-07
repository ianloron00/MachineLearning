import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math 
import random
import scipy.io as spio
import collections

columns = ['coluna 1', 'coluna 2']
train = pd.read_csv("cluster.dat", decimal=".", sep=' ', names=columns)
eps = 0.005
minpts = 10

def normalize(data):
  normData = data.copy()
  for i in range(data.shape[1]):
    normData.iloc[:,i] /= np.linalg.norm(data.iloc[:,i])
  return normData

class CustomDBSCAN:
  def __init__(self):
    # 0 - undef, -1 - noise, -2 - border, -3 - core
    self.label = 0

# find the close points of the current point analysed. (Euclidian referential)
def findNeighbors(data, curr_point, radius):
  points = []
  for neighbor in range(len(data)):
    if np.linalg.norm(data.iloc[neighbor]-data.iloc[curr_point]) <= radius:
      points.append(neighbor)   
  return points

# Expand Clusters, starting from core points
def ExpandClusters(data, adj_list, point_label, c, epsilon, newCluster):

  for p in adj_list[c]:
    # if it is a new point
    if point_label[p] != -2:  
      newCluster = ExpandClusters(data, adj_list, point_label, p, epsilon, newCluster)
      newCluster = np.append(newCluster, p)
      point_label[p] = -2
  return newCluster


def effDBSCAN(data, epsilon, minP):
  N = []
  S = [[]]
  core_points = []
  C = 0
  point_label = np.zeros(len(data),dtype=int) 
  adj_list = []
  # adj_list = point_label.copy()
  
  # 1- find adjacent list of points, and find the core points.
  for p in range(len(data)):
    if point_label[p] != 0: continue
    N = findNeighbors(data, p, epsilon)
    if len(N) >= minP:
      point_label[p] = -3
      core_points.append(p)
    else:
      point_label[p] = -1
    adj_list.append(N)

  print('adj list:')  
  for lin in range(len(adj_list)):
    print(str(lin)+': ' + adj_list[lin])
  # 2- Expand Clusters, starting from the core.
  for c in core_points:
    if point_label[c] == -3: # may change during ExpandClusters
      C=C+1
      point_label[c] = C
      S.append([p])
    S = ExpandClusters(data, adj_list, point_label, c, epsilon, S[C ])

  for o in range(len(point_label)):
    if point_label[o] > -2:
      S[0] = np.append(S[0],o)
  return S
  # print('Cluster '+str(p)+' (size '+str(len(S[C]))+'): '+str(S[C]))

#Function to plot final result
def plotRes(train, data, main_title):
  dicColors = {0:'black', 1:'orange', 2:'purple', 3:'red', 4:'blue', 5:'green', 
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
  return effDBSCAN(train_data,eps,minpts)

def callPlot(train_data,pointlabel, main_title):
  cl = len(pointlabel)
  for i in range (len(pointlabel)):
    print("cluster "+str(i)+": "+str(pointlabel[i]))
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
          'TRIAL Clusters division applying method DBSCAN (epsilon 0.05) - data from cluster.dat')

main()