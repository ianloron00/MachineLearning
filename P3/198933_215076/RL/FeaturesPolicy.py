from BFSPolicy import *
from game import Directions, Actions
import util

class Extractor:
    def getFeatures(self,state,action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # compute legal forward actions
        next_actions=Actions.getLegalNeighbors((next_x,next_y), walls)
        # scared_ghosts_pos_1_step=[Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) >0]
        # active_ghosts_pos_1_step=[Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) <=0]
        
        # if scared_ghosts_pos_1_step!=[]:
        #     print("active ghosts_ {}\nscared ghosts_ {}".format(
        #           active_ghosts_pos_1_step,scared_ghosts_pos_1_step))

        # older - has problems
        # active_ghosts_pos_2_steps=[Actions.getLegalNeighbors(active_ghosts_pos_1_step[g][p], walls) for g in range(len(ghosts)) for p in range(len(active_ghosts_pos_1_step)) if ghostScaredTime(g+1,state) <=0]
        # scared_ghosts_pos_2_steps=[Actions.getLegalNeighbors(scared_ghosts_pos_1_step[g][p], walls) for g in range(len(ghosts)) for p in range(len(scared_ghosts_pos_1_step)) if ghostScaredTime(g+1,state) >0]

        # newer
        # active_ghosts_pos_2_steps=[Actions.getLegalNeighbors(p, walls) for g in active_ghosts_pos_1_step for p in g]
        # scared_ghosts_pos_2_steps=[Actions.getLegalNeighbors(p, walls) for g in scared_ghosts_pos_1_step for p in g]
        
        # if scared_ghosts_pos_1_step!=[]:
        #     print("active ghosts_2_ {}\nscared ghosts_2_ {}".format(
        #           active_ghosts_pos_2_steps,scared_ghosts_pos_2_steps))



        # finds if there is closed scared ghosts: 
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) >0)
        # features["#-of-scared-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(q, walls) for a in next_actions for p in scared_ghosts_pos_2_steps for q in p)
        features["#-of-scared-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions for g in range(len(ghosts)) if ghostScaredTime(g+1,state) >0)

        # finds if there is activ ghost nearby:
        features["#-of-active-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) <=0)
        # features["#-of-active-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(q, walls) for a in next_actions for p in active_ghosts_pos_2_steps for q in p)
        features["#-of-active-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions for g in range(len(ghosts)) if ghostScaredTime(g+1,state) <=0)

        # if not features["#-of-active-ghosts-1-step-away"] and bool((next_x,next_y) in state.getCapsules()):
        #     features['has-capsule-1-step-away']=1.0
        
        if not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        fruit = Graph.getClosestPos((next_x, next_y), walls, food)
        if fruit is not None:
            dist=fruit.dist
            dir=fruit.dir
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            # features["run-to-catch-closest-fruit"]=float(int(action==dir)/(dist+1))

        scared_ghosts=[ghosts[s] for s in range(len(ghosts)) if ghostScaredTime(s+1,state)>0]
        scared_ghost=Graph.getClosestPos((next_x, next_y), walls, scared_ghosts)
        if scared_ghost != None:
            scared_ghost_dist=scared_ghost.dist
            scared_ghost_index=scared_ghost.ghost_id 
            scared_ghost_dir=scared_ghost.dir
            features["scaredTime-ghostDist"] = float(ghostScaredTime(scared_ghost_index+1,state))*0.5 - float(scared_ghost_dist) / (walls.width * walls.height)
            features["run-to-catch-scared-ghost"]=float(int(action==scared_ghost_dir)/(
                scared_ghost_dist+1))*0.5
            # print("catch closest fruit {}".format(features["run-to-catch-closest-fruit"]))
            # print("run-to-catch-scared-ghost{}".format(features["run-to-catch-scared-ghost"]))
        
            # if (features["#-of-active-ghosts-2-steps-away"] or scared_ghost_dist==2):
            #     comparison=[a in Actions.getLegalNeighbors(ghosts[g], walls) 
            #                 for a in next_actions for g in range(len(ghosts)) 
            #                 if ghostScaredTime(g+1,state) >0]
            #     pos_ghosts=[Actions.getLegalNeighbors(ghosts[g], walls) 
            #                 for g in range(len(ghosts)) 
            #                 if ghostScaredTime(g+1,state) >0]
            #     print("_action {}, 1-step away {}".format((next_x,next_y), features["#-of-active-ghosts-1-step-away"]))
            #     print("next_action {}, 2-steps away {}".format(next_actions, features["#-of-active-ghosts-2-steps-away"]))
            #     print("boolean comparison {}".format(comparison))
            #     print("next_actions {}; ghost_positions {}".format(next_actions, pos_ghosts))
        features.divideAll(10.0)

        return features

def ghostScaredTime(index, state):
    return state.getGhostState(index).scaredTimer
