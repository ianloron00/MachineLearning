# import os
import numpy as np
from pacman import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    numTraining=3
    numGames=numTraining+7
    agentargs=[]
    """
    número de passos máximo é definido em __init__ de Game
    """
    # alpha=0.15,gamma=0.1,epsilon=0.1
    agentargs.append('alpha=0.55,gamma=0.9,epsilon=0.6,numTraining={},numGames={}'.format(
        numTraining,numGames))
    
    for i in agentargs:
        # 'ApproximateQPolicy' or 'QLearning'
        args = readCommand(['-a',i,'-l','originalClassic','-p','ApproximateQPolicy','-z',2,
                            '-x',str(numTraining),'-n',str(numGames)])
        game = runGames(**args)
        
        scores = [k.state.getScore() for k in game]
        plt.plot(np.arange(len(scores)) + 1, scores)    
        plt.title(str(i))
        plt.show()
