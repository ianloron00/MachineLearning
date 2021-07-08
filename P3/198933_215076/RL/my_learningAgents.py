# python2 pacman.py -l smallClassic -z 2 -p QLearning

"""from typing import Counter
from game import Directions, Game 
import util
from game import Agent """
from __future__ import division
from pacman import Directions
from game import Agent
# from featureExtractors import *
from game import Actions

import os
import pickle
import time
import random
import math
import game
import util                

# Vertices
class Vertex:
    def __init__(self,pos):
        self.dir=None
        self.dist=0
        self.pos=pos
        # specific - ghosts
        self.ghost_id=None
        # internal - vetor V
        self.index=0
        # internal - getClosestPos
        self.parent=None

class Graph:
    @staticmethod
    def getClosestPos(pos,walls,map):
        
        V,u,pacman=Graph.BFS(pos,walls,map)
        if u==None: return None
       
        v=u
        while v.parent!=pacman.index: 
            dist=v.dist
            v_pos=v.pos
            v=V[v.parent]
            v.dist=dist
            v.pos=v_pos
        return v

    @staticmethod
    def BFS(pos,walls,map):
        p=Vertex(pos)
        queue=[p]
        V=[p]
        visited=[pos]
        while queue!=[]:
            u=queue.pop(0)
            for new_pos in Actions.getLegalNeighbors(u.pos,walls):
                if new_pos not in visited:
                    visited.append(new_pos)
                    v=Vertex(new_pos)
                    v.index=len(V)
                    v.parent=u.index
                    v.dist=u.dist+1
                    v.dir=Actions.vectorToDirection((v.pos[0]-u.pos[0],v.pos[1]-u.pos[1]))
                    x,y=new_pos
                    isPos=False
                    # fruits
                    if type(map[0])==list: isPos=map[x][y]
                    # ghosts
                    else: isPos=(x,y) in map
                    V.append(v)                    
                    if isPos:
                        return V,v,p
                    else:
                        queue.append(v)
        return (None,None,None)
    

class myQLearning(Agent):
    REWARD_LOSE=-10 # -300
    def __init__(self,alpha=0.2,gamma=0.95, epsilon=0.5):
        self.alpha=float(alpha)
        self.gamma=float(gamma)
        self.epsilon=float(epsilon)
        self.reward=0
        self.score=0
        self.last_state=None
        self.new_state=[]
        self.last_action=None
        self.legal_actions=None
        self.new_legal_actions=None
        self.iteration=1
        self.Q=util.Counter()

    def update_Q_table(self):
        
        a=self.last_action
        legal_actions=self.legal_actions
        new_legal_actions=self.new_legal_actions
        s=self.last_state
        r=self.reward
        next_s=self.new_state

        if self.new_legal_actions==[]:
            next_Q=0
        else: 
            next_Q=max([self.Q[(next_s,i)] for i in new_legal_actions])
            # if a==Directions.STOP:
            #     print("in state {} there is: ".format(s))
            #     for all in legal_actions:
            #         print("(acao: {}, Q: {})\n".format(all,self.Q[(s, all)]))            
        self.Q[(s,a)]+=self.alpha*(r + self.gamma*next_Q - self.Q[(s,a)])
        
    """
        # Expected-Sarsa implementation
        prev_action=self.last_action
        legal_actions=self.legal_actions
        new_legal_actions=self.new_legal_actions
        prev_state=self.last_state
        r=self.reward
        next_state=self.new_state
        num_actions=len(legal_actions)
        num_new_actions=len(new_legal_actions)
        reward=self.reward

        predict = self.Q[(prev_state, prev_action)]
 
        expected_q = 0
        # q_max = np.max(self.Q[next_state, :])
        if num_new_actions:
            q_max=max([self.Q[(next_state,next_a)] for next_a in new_legal_actions])
        else:
            return
        greedy_actions = 0
        for i in new_legal_actions:
            if self.Q[(next_state,i)] == q_max:
                greedy_actions += 1
    
        non_greedy_action_probability=self.epsilon/num_actions
        greedy_action_probability=((1 - self.epsilon)/greedy_actions) + non_greedy_action_probability
 
        for i in new_legal_actions:
            if self.Q[(next_state,i)] == q_max:
                expected_q += self.Q[(next_state,i)] * greedy_action_probability
            else:
                expected_q += self.Q[(next_state,i)] * non_greedy_action_probability
 
        target = reward + self.gamma * expected_q
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)
        """

    def getState(self,state):
        new_state=[]
        pacmanPos=state.getPacmanPosition()
        fruitsPos=state.getFood()
        walls=state.getWalls()
        ghosts=state.getGhostPositions()
        ghostStates=state.getGhostStates()
        fruitsNum=state.getNumFood()
        agentsNum=state.getNumAgents()
        ghost_bool_states=[g.scaredTimer for g in ghostStates]
        list_scared_ghosts=[g for g in range(1,len(ghostStates)+1) 
                            if state.getGhostState(g).scaredTimer]
        capsules=state.getCapsules()

        # new_state.append(len(capsules)) 
        # new_state.append(pacmanPos)
        # new_state.append(tuple(ghost_bool_states))
        v_f=Graph.getClosestPos(pacmanPos,walls,fruitsPos)
        v_g=Graph.getClosestPos(pacmanPos,walls,ghosts)
        
        new_state.append(v_f.dir)
        new_state.append(v_f.dist//3)
        if v_g != None:
            v_g.ghost_id=ghosts.index(v_g.pos)
            new_state.append(v_g.dir)
            new_state.append(v_g.dist//3)
        # if list_scared_ghosts!=[]:
        #     scared_ghosts=[] 
        #     for i in list_scared_ghosts:
        #         scared_ghosts.append(ghosts[i-1])
# 
        #     v_sg=Graph.getClosestPos(pacmanPos,walls,scared_ghosts)
        #     if v_sg != None:
        #         v_sg.ghost_id=ghosts.index(v_sg.pos)
        #         new_state.append(v_sg.dir)
        #         new_state.append(v_sg.dist)

        return tuple(new_state)
    
    def getReward(self,state):
        reward=0
        x,y=state.getPacmanPosition()
        if state.hasFood(x,y):
            reward+=1
        if state.isWin():
            reward+=30
        if state.isLose():
            reward-=20
        if self.last_action==Directions.STOP:
            reward-=1
        return reward


    def initState(self,state):
        self.legal_actions=state.getLegalPacmanActions()
        if self.last_action==Directions.STOP and len(self.last_action)>1:
            self.legal_actions.remove(Directions.STOP)
        self.last_state=self.getState(state)

    def step(self,state,a):
        new_state=state.generateSuccessor(0, a)
        self.new_legal_actions=new_state.getLegalPacmanActions() 
        # set new_state
        self.new_state=self.getState(new_state)
        # update reward   
        self.reward= self.getReward(state) # state.getScore() - self.score

    def epsilon_greedy_policy(self,state,epsilon):
        if random.uniform(0,1)<epsilon:
            return random.choice(self.legal_actions)
        else:
            """ return max Q[s,a'] value, given a' in action space """
            return max(self.legal_actions, key=lambda a: self.Q[(state,a)] )
    

    def getAction(self, state):
        self.alpha=0.3 # max(0.1,1/(self.iteration/100 + 1))
        self.epsilon=max(1/(self.iteration/300 + 1),0.1)

        self.initState(state)
        self.last_action=self.epsilon_greedy_policy(self.last_state,self.epsilon)
        self.step(state, self.last_action)

        # if self.iteration>1000:
        #     self.epsilon=0
        #     self.alpha=0

        self.update_Q_table() 

        self.score=state.getScore()
        # if game is over.
        if self.reward<self.REWARD_LOSE or state.getNumFood()==0:
            self.score=0
            self.iteration+=1 
            
            if not self.iteration%100:
                print("state: {}".format(self.last_state))
                print("alpha: {}, gamma: {}, eps: {}\n iter: {}, reward: {}, numFood: {}\n".format(
                    self.alpha,self.gamma,self.epsilon,self.iteration,self.reward,state.getNumFood()
                ))
        
            


        return self.last_action



#======================================================================================
class myOldQLearning(Agent):

    def __init__(self,alpha=0.2,gamma=0.8,epsilon=0.5):
        self.alpha=float(alpha)
        self.gamma=float(gamma)
        self.epsilon=float(epsilon)
        self.reward=0
        self.score=0
        self.last_state=None
        self.new_state=[]
        self.legal_actions=None
        self.new_legal_actions=None
        self.iteration=0
        self.Q=util.Counter()

    def update_Q_table(self,_, s, a, r, next_s):
        # legal actions deste ou do outro estado?
        if self.legal_actions==[]:
            next_Q=0
        else: 
            next_Q=max([self.Q[(next_s,i)] for i in self.legal_actions])
            if next_Q != 0:
                print (next_Q)
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
        # created si that suimmulates pacmans's movements.
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
        pacmanPos=state.getPacmanPosition()
        fruitsPos=state.getFood()
        new_state.append(pacmanPos)
        new_state.append(self.findClosestFruit(pacmanPos,fruitsPos))
        new_state.append(tuple(state.getCapsules()))
        new_state.append(tuple(state.getGhostPositions()))
        new_state.append(state.getNumFood())
        new_state.append(state.getNumAgents())
        new_state.append(state.getScore())
        new_state.append(fruitsPos)
        return tuple(new_state)

    def epsilon_greedy_policy(self,_,state,epsilon):
        if random.uniform(0,1)<epsilon:
            return random.choice(self.legal_actions)
        else:
            """ return max Q[s,a'] value, given a' in action space """
            return max(self.legal_actions, key=lambda a: self.Q[(state,a)] )
    
    def getAction(self, state):
        self.iteration+=1
        self.alpha=4/self.iteration+3
        self.epsilon=1/self.iteration



        # self.reward=state.getScore()-self.score
        
        # self.new_state=state # self.getState(state)
        self.last_state=state
        self.legal_actions=self.last_state.getLegalPacmanActions() 
        if len(self.legal_actions)>1:
            self.legal_actions.remove(Directions.STOP)   

        self.last_action=self.epsilon_greedy_policy(self,self.last_state,self.epsilon)
        self.score=state.getScore()
        
        self.new_state=state.generateSuccessor(0, self.last_action)

        # self.legal_actions= state.getLegalPacmanActions()    
        self.legal_actions=self.new_state.getLegalPacmanActions() 
        if len(self.legal_actions)>1:
            self.legal_actions.remove(Directions.STOP)     

        self.reward=self.new_state.getScore() - self.score
        """      
        Atualiza a tabela-Q, com dados da interacao anterior:                  
        self,last_state,old_action,old_reward,new_state 
        """
        self.update_Q_table(self, self.last_state, 
                            self.last_action,self.reward, self.new_state)        
        

        # self.last_state=self.new_state
        # self.score=state.getScore()
        # self.last_action=a


        return self.last_action

#======================================================================================

class myDoubleQLearning(Agent):

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
        # not sure...
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
        self.legal_actions= state.getLegalPacmanActions() 

        Q1=self.Qb
        Q2=self.Qa
        if self.count%2:
            Q1=self.Qa
            Q2=self.Qb

        if self.count: 
            """      
            Atualiza a tabela-Q, com dados da interacao anterior:                  
                                self,last_state,        old_action,new(?)_reward,new_state 
            """
            self.update_Q_table(self,Q1,Q2,self.last_state, self.last_action,self.reward, state)

        self.last_state=state
        
        Q=Q1
        a=self.epsilon_greedy_policy(self,Q,state,self.epsilon)
        self.count+=1
        self.last_action=a
        return a