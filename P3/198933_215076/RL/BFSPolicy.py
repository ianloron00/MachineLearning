from game import Actions,Grid

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
        # if ghost
        if type(map[0])!=list:
            v.ghost_id=map.index(v.pos)
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
                    if type(map)==Grid: 
                        isPos=map[x][y]
                    # ghosts
                    else: isPos=(x,y) in map
                    V.append(v)                    
                    if isPos:
                        return V,v,p
                    else:
                        queue.append(v)
        return (None,None,None)
