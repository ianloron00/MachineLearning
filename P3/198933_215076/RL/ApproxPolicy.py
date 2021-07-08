from game import *
# from LearningPolicy import *
from QLPolicy import *
from FeaturesPolicy import *
from RW_files import *
import random,util,math,sys
from game import Agent 

# PacmanQAgent # QLAgent 
myAgent= QLAgent

class ApproximateQPolicy(myAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite 'getQValue'
       and 'update'.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='Extractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        myAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.write_to_file=False
        self.read_from_file=False
        if CAN_READ_WRITE_FILES:
            self.write_to_file = None
            self.read_from_file = None
        
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q_val = 0
        features = self.featExtractor.getFeatures(state, action)

        if self.read_from_file == None:
            self.read_from_file=checkIfRead()
          
        if self.read_from_file == True:
            dict = read_file(state)
            weights = util.Counter()
            for f in dict: weights[f] = dict[f]
            self.weights = weights
            self.read_from_file = False

        for f in features:
          q_val+=self.weights[f]*features[f]

        return q_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        diff = (reward + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state, action)

        features = self.featExtractor.getFeatures(state, action) 
        for f in features:
          self.weights[f] += self.alpha*diff*features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        myAgent.final(self, state)
        if not self.episodesSoFar%10:
            print("number iterations processed so far: {}".format(self.episodesSoFar))

        if self.episodesSoFar == self.numGames and self.write_to_file == None:
            print ("\nFeature weights: {}\n".format(self.weights))  
            self.write_to_file=checkIfSave()
            
        if self.write_to_file:
          save_file(state, self.weights)


    
