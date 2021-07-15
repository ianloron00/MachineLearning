from game import Directions, Agent, Actions
import random,util,time
import matplotlib.pyplot as plt
import numpy as np

class QLAgent(Agent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self,alpha=0.2,gamma=0.8,epsilon=0.5,numTraining=0,numGames=10,**args):
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.episodeRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.numGames=int(numGames)
        if self.numTraining:
            self.eps_rate=(self.discount/self.numTraining)
            self.alpha_rate=(self.alpha/self.numTraining)
        else:
            self.eps_rate=0
            self.alpha_rate=0
        self.reward=0
        self.score=0
        self.last_state=None
        self.new_state=[]
        self.Q=util.Counter()
        self.n_steps=0
        self.n_cumulative_actions=[]
        self.cumulative_score=[]
        self.n_wins=0
        self.cumulative_wins=[]

    def getQValue(self, state, action):
        ans=self.Q[(state,action)]
        if ans == None: return 0.0
        return ans

    def computeValueFromQValues(self, state):
        action = self.computeActionFromQValues(state)
        if action != None:
          return self.getQValue(state,action)
        return 0.0

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def computeActionFromQValues(self, state):        
        legalActions =  state.getLegalActions()
        max_action = None
        max_value = -100000
        for a in legalActions:
          cur_val = self.getQValue(state, a)
          if max_value <= cur_val:
            max_value = cur_val
            max_action = a

        return max_action
    
    def getAction(self, state):
        # Pick Action
        self.step+=1
        legalActions = state.getLegalActions()  
        action = None
        if len(legalActions)>0:
          explore = util.flipCoin(self.epsilon)
          if explore == True:
            action = random.choice(legalActions)
          else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        best_Q=self.computeValueFromQValues(nextState)
        self.Q[state,action]=((1 - self.alpha)*self.Q[state,action] + 
                                self.alpha*(reward + self.discount*best_Q))

    def epsilon_greedy_policy(self,state):
        legalActions = state.getLegalActions()
        action = None
        if len(legalActions)>0:
            explore = util.flipCoin(self.epsilon)
        if explore == True:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.last_state = state
        self.lastAction = action
    
    def getAction(self, state):
        self.legal_actions=state.getLegalActions()
        action = self.epsilon_greedy_policy(state)
        self.doAction(state,action)
        self.n_steps+=1
        return action

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        if not self.last_state is None:
            deltaReward = (state.getScore() - self.last_state.getScore())
            self.observeTransition(self.last_state, self.lastAction, state, deltaReward)
        self.stopEpisode()

        if self.epsilon>0:
            self.epsilon-=self.eps_rate
        
        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print ('%s\n%s' % (msg,'-' * len(msg)))
            self.epsilon=0
            self.alpha=0

        elif self.episodesSoFar > self.numTraining:
            self.n_cumulative_actions.append(self.n_steps)
            self.cumulative_score.append(state.getScore())
            if state.isWin():
                self.n_wins+=1
            self.cumulative_wins.append(state.isWin())

            if self.episodesSoFar==self.numGames:
                # print("rewards: {}".format(self.cumulative_score))
                # print("wins: {}".format(self.cumulative_wins))
                # print("number of actions per episode: {}".format(self.n_cumulative_actions))
                print("median rewards: {}".format(np.median(self.cumulative_score)))
                print("win rate: {}".format(float(self.n_wins/len(self.cumulative_wins))) )
                
                plt.plot(np.arange(len(self.n_cumulative_actions)) + 1, self.n_cumulative_actions)    
                plt.title("number of actions per episode")
                plt.show()

        self.n_steps=0

        
    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)
    
    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.last_state = None
        self.lastAction = None
        self.episodeRewards = 0.0
    
    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning
    
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.last_state is None:
            reward = state.getScore() - self.last_state.getScore()
            self.observeTransition(self.last_state, self.lastAction, state, reward)
        return state

    