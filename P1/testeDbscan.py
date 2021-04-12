import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

cabecalho = ['coluna 1', 'coluna 2']
dataset = pd.read_csv("cluster.dat", decimal=".", sep=' ', names=cabecalho)
eps = 0.005
minpts = 10

#Classe Dbscan
class Dbscan():

  #Inicializando atributos da classe
  def __init__(self, train, epsilon=1, min_points=5):
      self.train = np.array(train)
      self.epsilon = epsilon
      self.min_points = min_points
      self.cluster = 0
      self.noise = 0

  #Calculo da distancia entre 2 pontos
  def dist(self, pt1, pt2):
      return np.linalg.norm(pt1 - pt2)

  # Pega todos os vizinhos de v1 nos dados de treino que estao dentro do raio epsilon
  def getNeighbors(self, v1):
      neighbors = []
      for i in range(len(self.train)):
          v2 = self.train[i, :v1.shape[0]]
          if self.dist(v1, v2) <= self.epsilon:
              neighbors.append((v2,i))
      return neighbors

  # Metodo para treinar o modelo
  def fit(self):
      #Tamanho do dataset
      l, c = self.train.shape
      #Adicionando uma nova coluna dentro dos dados de treino para controlar a que cluster pertence
      self.train = np.append(self.train, np.array([[-1] * len(self.train)]).T, axis=1)

      for i in range(len(self.train)):
          # Verifica que o ponto ja esta atribuido a algum cluster
          if self.train[i, c] != -1:
              continue
          neighbors = self.getNeighbors(self.train[i, :c])
          # Caso o ponto nao tenha vizinho o suficiente ele é atribuido como ruido
          if len(neighbors) < self.min_points:
              self.train[i, c] = self.noise
              continue

          self.cluster += 1
          #Atribui o ponto a um cluster
          self.train[i, c] = self.cluster
          neighbors_vistos = []

          # Verifica qual ponto ja foi visitado para que nao se repita o processo
          for j in neighbors:
              neighbors_vistos.append(j[1])


          while len(neighbors) > 0:

              # Pega o ultimo ponto da lista de vizinhos e logo em seguida remove ele da lista
              atual = neighbors[len(neighbors)-1]
              neighbors = neighbors[0:len(neighbors)-1]

              #Atribui ao ponto um cluster
              self.train[atual[1],c] = self.cluster

              #Pega todos os vizinho deste novo ponto, para verificar se farao parte do mesmo cluster
              # ou seja, se sao vizinhos do vizinho
              neighbors2 = self.getNeighbors(self.train[atual[1],:c])

              # Verifica se esse ponto tem uma quantidade boa o suficiente de vizinho para
              # serem adicionados ao cluster atual
              if len(neighbors2) >= self.min_points:

                  # Para cada vizinho do novo ponto se ele ainda nao foi visitado
                  # adiciona ele na lista de vizinhos e na lista de vizinhos visitados
                  for j in neighbors2:
                      if not (j[1] in neighbors_vistos):
                          neighbors_vistos.append(j[1])
                          neighbors = [j] + neighbors
  # Metodo para tentar predizer a que cluster pertence um conjunto de teste
  def predict(self, test):
      ret = []
      for i in range(len(test)):
          l, c = self.train.shape
          # A ideia aqui é pegar o ponto novo e verificar quais seriam os vizinhos dele
          # nos dados de treino
          neighbors = self.getNeighbors(np.asarray(test.iloc[i,:]))

          # Se ele teria algum vizinho, eu atribuio o ponto ao mesmo cluster do primeiro vizinho
          # Senao coloco ele como outlier, ou seja, -1
          if len(neighbors) > 0:
              label = self.train[neighbors[0][1], c-1]
          else:
              label = -1.0
          ret.append(label)

      return ret


def normalize(data):
  normData = data.copy()
  for i in range(data.shape[1]):
    normData.iloc[:,i] /= np.linalg.norm(data.iloc[:,i])
  return normData

# Pensar melhor em relação a esse metodo
# o que estou fazendo no momento é embaralhar os dados e pegar o p*len(data) primeiros
# e atribuir para os dados de treino e os (1-p)*len(data) ultimos atribuindo para os dados de teste
def split(data, p=0.9):
    data_copy = data.iloc[np.random.permutation(len(data))]
    return data_copy[0:int(data_copy.shape[0]*p)], data_copy[int(data.shape[0]*p):]

def plotGraphic(X_cl) :
  # Tentando prever para que cluster pertence
  # X_cl = db.predict(test)
  print(X_cl)
  dicionarioCores = {-1:'purple',0 : 'red', 1 : 'blue', 2: 'green', 3: 'pink', 4: 'yellow'}
  label_color = [dicionarioCores[l] for l in X_cl]

  c1 = 0
  c2 = 1
  labels = ['x', 'y']
  c1label = labels[c1]
  c2label = labels[c2]
  title = c1label + ' x ' + c2label

  plt.figure(figsize = (12,12))
  plt.scatter(test.iloc[:,c1],test.iloc[:, c2], c=label_color, alpha=0.3)
  plt.xlabel(c1label, fontsize=18)
  plt.ylabel(c2label, fontsize=18)
  plt.suptitle(title, fontsize=20)
  plt.savefig(title + '.png')
  plt.show()

  print(datetime.datetime.now() - begin_time)

# Calculo do tempo do algoritmo
begin_time = datetime.datetime.now()

#Normalizando os dados, para que nao de problema no calculo de distancia
normTrain = normalize(dataset)

#Metodo teste para dividir os dados de treino e os dados de teste -> tera P dos dados em treino
train,test = split(normTrain, p=0.9)

db = Dbscan(train, eps, minpts)

#Treinando com os dados de treino
db.fit()

# X_cl = train # db.fit()
X_cl = train.iloc[1:,:]
plotGraphic(X_cl)

X_cl = db.predict(test)
plotGraphic(X_cl)