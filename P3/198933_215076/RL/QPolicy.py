# python pacman.py -l tinyMaze -z 2 --pacman DumbAgent
from BFSPolicy import Graph
from typing import Counter
from game import Directions, Game 
import util
import numpy as np
import random
from game import Agent,Actions
import matplotlib.pyplot as plt

class QLearning(Agent):

    reward=0
    score=0
    old_state=None
    new_state=[]
    Q=util.Counter()

    def __init__(self,alpha=0.7,gamma=0.9,epsilon=0.5,numTraining=100,numGames=110):
        self.alpha=float(alpha)
        self.gamma=float(gamma)
        self.epsilon=float(epsilon)
        self.numTraining=int(numTraining)
        if self.numTraining:
            self.eps_rate=(self.epsilon/self.numTraining)
            self.alpha_rate=(self.alpha/self.numTraining)
        else:
            self.eps_rate=0
            self.alpha_rate=0
        self.n_steps=0
        self.n_cumulative_actions=[]
        self.episodesSoFar=0
        self.numGames=int(numGames)
        self.action=Directions.STOP
        self.cumulative_Reward=[]
        self.n_wins=0
        self.cumulative_wins=[]

    def update_Q_table(self,_, s, a, r, next_s):
        next_Q=max([self.Q[(next_s,i)] for i in self.legal_actions])
        self.Q[(s,a)]+=self.alpha*(r + self.gamma*next_Q - self.Q[(s,a)])

    def hasFruit(self, position, fruits):
        return fruits[position[0][0]][position[0][1]]

    def findClosestFruit(self,position, fruits):
        # board sizes
        y=fruits.height 
        x=fruits.width
        # pacman's position 
        yp=position[1]
        xp=position[0]
        k=1
        # as pacman does not move in diagonals, it was
        # created this cond. that simmulates pacmans's movements.
        while k<=x:
            for dx in range(0,k+1):
                dy=k-dx
                i=xp+dx
                if i<x:
                    j=yp+dy
                    position=[(i,j)]
                    if j<y and self.hasFruit(position,fruits):
                        return tuple(position)
                    j=yp-dy
                    position=[(i,j)]
                    if j>=0 and self.hasFruit(position,fruits):
                        return tuple(position)
                if not dx: continue
                i=xp-dx
                if i>=0:
                    j=yp+dy
                    position=[(i,j)]
                    if j<y and self.hasFruit(position,fruits):
                        return tuple(position)
                    j=yp-dy
                    position=[(i,j)]
                    if j>=0 and self.hasFruit(position,fruits):
                        return tuple(position)
            k+=1

    def getState(self,state):
        new_state=[]
        # pacmanPos=state.getPacmanPosition()
        # fruitsPos=state.getFood()
        # ghostsPos=state.getGhostPositions()
        # walls=state.getWalls()
        # new_state.append(pacmanPos)
        # new_state.append(self.findClosestFruit(pacmanPos,fruitsPos))
        # new_state.append(tuple(state.getCapsules()))
        # new_state.append(tuple(state.getGhostPositions()))
        # new_state.append(state.getNumFood())
        # new_state.append(state.getNumAgents())
        # new_state.append(state.getScore())

        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(self.action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # compute legal forward actions
        next_actions=[Actions.getLegalNeighbors((next_x,next_y), walls)]
        
        # finds if there is closed scared ghosts 1 and 2 steps away: 
        n_scared_ghosts_1_step_away = sum((next_x, next_y) in 
                Actions.getLegalNeighbors(ghosts[g], walls) 
                for g in range(len(ghosts)) 
                if ghostScaredTime(g+1,state) >0)
        n_scared_ghosts_2_steps_away = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) 
                                    for a in next_actions for g in range(len(ghosts)) 
                                    if ghostScaredTime(g+1,state) >0)

        # finds if there is activ ghost nearby:
        n_active_ghosts_1_step_away = sum((next_x, next_y) in 
                                    Actions.getLegalNeighbors(ghosts[g], walls) 
                                    for g in range(len(ghosts)) 
                                    if ghostScaredTime(g+1,state) <= 0)
        n_active_ghosts_2_steps_away = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) 
                                        for a in next_actions for g in range(len(ghosts)) 
                                        if ghostScaredTime(g+1,state) <=0)

        has_capsule=False
        # if there is no danger nearby, reward if pacman eats capsule:
        if not n_active_ghosts_1_step_away and bool((next_x,next_y) in state.getCapsules()):
            has_capsule=True
        
        eats_food=False
        # if there is no danger nearby, reward if pacman eats fruit:
        if (not n_active_ghosts_1_step_away 
            and not n_active_ghosts_2_steps_away and food[next_x][next_y]):
            eats_food = True
        
        # find closest food:
        fruit = Graph.getClosestPos((next_x, next_y), walls, food)
        if fruit is not None:
            dist=fruit.dist
            dir=fruit.dir
            closest_food = float(dist) 
            food_direction=dir==self.action
        
        # find closest scared ghost:
        scared_ghosts=[ghosts[s] for s in range(len(ghosts)) if ghostScaredTime(s+1,state)>0]
        scared_ghost=Graph.getClosestPos((next_x, next_y), walls, scared_ghosts)
        if scared_ghost != None:
            scared_ghost_dist=scared_ghost.dist
            scared_ghost_index=scared_ghost.ghost_id 
            scared_ghost_dir=scared_ghost.dir
            scaredTime_ghostDist = (float(ghostScaredTime(scared_ghost_index+1,state)) - 
                                               float(scared_ghost_dist)) # / (walls.width * walls.height))
            run_to_catch_scared_ghost=self.action==scared_ghost_dir
            # (float(int(self.action==scared_ghost_dir)/(scared_ghost_dist+1))*0.5)

        new_state.append(n_scared_ghosts_1_step_away)
        new_state.append(n_scared_ghosts_2_steps_away)
        new_state.append(n_active_ghosts_1_step_away)
        new_state.append(n_active_ghosts_2_steps_away)
        new_state.append(has_capsule)
        new_state.append(eats_food)
        if fruit is not None:
            new_state.append(closest_food)
            new_state.append(food_direction)
        if scared_ghost is not None:
            new_state.append(scaredTime_ghostDist)
            new_state.append(run_to_catch_scared_ghost)

        return tuple(new_state)

    def epsilon_greedy_policy(self,_,state,epsilon):
        if random.uniform(0,1)<epsilon:
            return random.choice(self.legal_actions)
        else:
            """ return max Q[s,a'] value, given a' in action space """
            return max(self.legal_actions, key=lambda a: self.Q[(state,a)] )
    
    def getAction(self, state):

        self.reward=state.getScore()-self.score
        
        self.new_state=self.getState(state)

        if self.old_state!= None: 
            """      
            Atualiza a tabela-Q, com dados da interação anterior:                  
                                self,old_state,old_action,old_reward,new_state 
            """
            self.update_Q_table(self, self.old_state, 
                                self.action,self.reward, self.new_state)        
        
        self.old_state=self.new_state
        self.score=state.getScore()
        
        self.legal_actions= state.getLegalPacmanActions()    
        a=self.epsilon_greedy_policy(self,self.new_state,self.epsilon)
        self.action=a
        self.n_steps+=1
        return a

    def final(self,state):
        self.episodesSoFar+=1
        if self.epsilon: 
            self.epsilon-=self.eps_rate
        # if self.alpha:
        #     self.alpha-=self.alpha_rate
        
        if not self.episodesSoFar%50:
            print("number iterations processed so far: {}".format(self.episodesSoFar))
            # print("num training, num games: {}, {}".format(self.numTraining,self.numGames))
            # print("score so far: {}".format(state.getScore()))
            # print("my state so far: {}".format(self.old_state))
            # print("hyperparameters: a,g,e__: {}, {}, {}_{}".format(self.alpha,self.gamma,self.epsilon,self.eps_rate))

        if self.episodesSoFar == self.numTraining:
            print("finished training (turning off epsilon and alpha)")
            self.alpha=0
            self.epsilon=0
            
        elif self.episodesSoFar > self.numTraining:
            # eps_game=self.episodesSoFar-self.numTraining
            # print("number of actions taken in game-episode[{}]: {}".format(eps_game, self.n_steps))
            self.n_cumulative_actions.append(self.n_steps)
            self.cumulative_Reward.append(state.getScore())
            if state.isWin():
                self.n_wins+=1
            self.cumulative_wins.append(state.isWin())
        
            if self.episodesSoFar==self.numGames:
                print("median rewards: {}".format(np.median(self.cumulative_Reward)))
                print("win rate: {}".format(self.n_wins))
                # print("number of actions per episode: {}".format(self.n_cumulative_actions))

                plt.plot(np.arange(len(self.n_cumulative_actions)) + 1, self.n_cumulative_actions)    
                plt.title("number of actions per episode")
                plt.show()

        self.n_steps=0
        
#====================================================================================

class DoubleQLearning(Agent):

    count=0
    reward=0
    observation=None
    Qa=util.Counter()
    Qb=util.Counter()

    def __init__(self,alpha=0.7,gamma=0.9,epsilon=0.2):
        self.alpha=float(alpha)
        self.gamma=float(gamma)
        self.epsilon=float(epsilon)

    def update_Q_table(self,_, Q1, Q2, s, a, r, next_s):
        a_max = max(self.legal_actions, key=lambda a: Q1[(s,a)])
        Q1[(s,a)]+=self.alpha*(r + self.gamma*Q2[(next_s,a_max)] - Q1[(s,a)])

    def epsilon_greedy_policy(self,_,Q1, state,epsilon):
        if random.uniform(0,1)<epsilon:
            return random.choice(self.legal_actions)
        else:
            """ return max Q[s,a'] value, given a' in action space """
            return max(self.legal_actions, key=lambda a: Q1[(state,a)] )
    
    def getAction(self, state):

        self.reward=state.getScore()

        Q1=self.Qb
        Q2=self.Qa
        if self.count%2:
            Q1=self.Qa
            Q2=self.Qb

        if self.count: 
            """      
            Atualiza a tabela-Q, com dados da interação anterior:                  
                                self,old_state,        old_action,new(?)_reward,new_state 
            """
            self.update_Q_table(self,Q1,Q2,self.old_state, self.action,self.reward, state)

        self.old_state=state
        
        self.legal_actions= state.getLegalPacmanActions() 
        Q=Q1
        a=self.epsilon_greedy_policy(self,Q,state,self.epsilon)
        self.count+=1
        self.action=a
        return a

def ghostScaredTime(index, state):
    return state.getGhostState(index).scaredTimer