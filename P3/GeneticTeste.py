import os
import numpy as np
from pacman import *
import matplotlib.pyplot as plt

def generateIndividuo(size=1000):
    list_move = ['d','a','s','w','q']
    str_ret = ""
    for i in range(size):
        str_ret += " " + random.choice(list_move)
    return str_ret.strip()

def generateFirstGeration(size=100,size_individuo=1000):
    ret = []
    for i in range(size):
        ret.append(generateIndividuo(size_individuo))
    return ret

def rouletteMethod(scores, i):
    arrays = np.arange(i)
    p = (scores-np.min(scores))/np.sum(scores-np.min(scores))
    return np.random.choice(arrays,p=p)

def selectMelhorIndividuo(scores, ind):
    ret = []
    s = np.sort(scores)
    s = s[len(s)-1]
    return np.asarray([ind[np.where(scores == s)[0][0]].split()])


# Reprodução por ponto duplo
def reproduction(scores, ind):
    i = ind[0].split()
    ini = random.randint(0, len(i))
    meio = random.randint(ini, len(i))

    casal1 = ind[rouletteMethod(scores,len(scores))].split()
    casal2 = ind[rouletteMethod(scores,len(scores))].split()

    filho1 = np.concatenate((casal1[0:ini],casal2[ini:meio]),axis=0)
    filho1 = np.concatenate((filho1, casal1[meio:]), axis=0)

    filho2 = np.concatenate((casal2[0:ini], casal1[ini:meio]), axis=0)
    filho2 = np.concatenate((filho2, casal2[meio:]), axis=0)

    return filho1, filho2

def voltarInd(ind):
    arr = []
    for i in ind:
        arr.append(" ".join(i))
    return arr

def novaGeracao(scores, ind):
    arr = selectMelhorIndividuo(scores,ind)
    while len(arr) < len(scores):
        f1,f2 = reproduction(scores, ind)

        arr = np.concatenate((arr,[f1]))
        arr = np.concatenate((arr, [f2]))
    # FAÇO AS MUTAÇÔES
    return voltarInd(arr)

def selectMelhorEPiorIndividuo(scores, ind):
    ret = []
    s = np.sort(scores)
    s = s[len(s)-1]
    s2 = np.sort(scores)[0]
    return np.where(scores == s)[0][0], np.asarray([ind[np.where(scores == s)[0][0]].split()]), np.where(scores == s2)[0][0], np.asarray([ind[np.where(scores == s2)[0][0]].split()])

if __name__ == '__main__':
    numGame = 30
    individuos = generateFirstGeration(11)
    ar = []
    melhorInd = []
    piorInd = []
    mediaInds = []

    for j in range(5):
        medias = []
        for i in individuos:
            args = readCommand(['-a', 'list_command='+i, '-p', 'TestAgents','-q','-n',str(numGame)])
            game = runGames(**args)
            scores = [k.state.getScore() for k in game]
            medias.append(sum(scores) / float(len(scores)))
        mep = selectMelhorEPiorIndividuo(medias, individuos)
        melhorInd.append(medias[mep[0]])
        piorInd.append(medias[mep[2]])
        mediaInds.append(np.mean(medias))
        individuos = novaGeracao(medias, individuos)
        ar.append({str(j+1):medias})
    plt.plot(np.arange(5)+1, melhorInd)
    plt.plot(np.arange(5) + 1, piorInd)
    plt.plot(np.arange(5) + 1, mediaInds)
    plt.title("5 Geraçoes")
    plt.show()


    for j in range(5):
        medias = []
        for i in individuos:
            args = readCommand(['-a', 'list_command='+i, '-p', 'TestAgents','-q','-n',str(numGame)])
            game = runGames(**args)
            scores = [k.state.getScore() for k in game]
            medias.append(sum(scores) / float(len(scores)))
        mep = selectMelhorEPiorIndividuo(medias, individuos)
        melhorInd.append(medias[mep[0]])
        piorInd.append(medias[mep[2]])
        mediaInds.append(np.mean(medias))
        individuos = novaGeracao(medias, individuos)
        ar.append({str(j+1):medias})
    plt.plot(np.arange(10)+1, melhorInd)
    plt.plot(np.arange(10) + 1, piorInd)
    plt.plot(np.arange(10) + 1, mediaInds)
    plt.title("10 Geraçoes")
    plt.show()

    for j in range(5):
        medias = []
        for i in individuos:
            args = readCommand(['-a', 'list_command=' + i, '-p', 'TestAgents', '-q', '-n', str(numGame)])
            game = runGames(**args)
            scores = [k.state.getScore() for k in game]
            medias.append(sum(scores) / float(len(scores)))
        mep = selectMelhorEPiorIndividuo(medias, individuos)
        melhorInd.append(medias[mep[0]])
        piorInd.append(medias[mep[2]])
        mediaInds.append(np.mean(medias))
        individuos = novaGeracao(medias, individuos)
        ar.append({str(j + 1): medias})
    plt.plot(np.arange(15) + 1, melhorInd)
    plt.plot(np.arange(15) + 1, piorInd)
    plt.plot(np.arange(15) + 1, mediaInds)
    plt.title("15 Geraçoes")
    plt.show()

    print(ar)