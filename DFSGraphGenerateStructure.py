import numpy as np
import collections
import copy
import sys
from scipy import io

class Graph():
    def __init__(self, vertices, start, end, limitCon):
        self.graph = collections.defaultdict(list) 
        self.V = vertices 
        self.start = start
        self.end = end
        self.limitCon = limitCon

    def addEdge(self, u, v): 
        self.graph[u].append(v) 
    
    def graphUnuque(self):
        for i in self.graph:
            self.graph[i] = (list(set(self.graph[i])))



    def generateStruct(self, g):
        visited = np.array([0] * self.V)
        elementsMask = [True] * self.V 
        elementsMask[self.start] = False 

        #for i in np.arange(1, 100):
        # count connections with input
        countCon = np.random.randint(1, self.limitCon + 1)
        # ids connections with input
        ids = np.unique([np.random.randint(self.limitCon + 1, self.V) for i in np.arange(countCon)]) 

        for j in ids:
                # add conns with input 
            g.addEdge(self.start, j)
            if visited[j] <= self.limitCon:
                self.genCon(g, j, visited, elementsMask)

    def genCon(self, g, v, visited, elementsMask):
        visited[v] += 1 
        elementsMask[v] = False
    
        if v == self.end:
            #if visited[v] <= self.limitCon:
                elementsMask[v] = True
                return
            #else:
                #return

        # count connections with input
        countCon = np.random.randint(1, self.limitCon + 1)
        # list elements for chousen
        posiblElems = np.arange(self.V)[elementsMask]
        #posiblElems = np.delete(posiblElems, v - 1)
        # ids connections with input
        ids = np.unique([np.random.choice(posiblElems) for i in np.arange(countCon)]) 

        for j in ids:
                # add conns with input 
                g.addEdge(v, j)
                if visited[j] <= self.limitCon:
                    self.genCon(g, j, visited, elementsMask)
                else:
                    return 1

        elementsMask[v] = True
        return 1
    def isCyclicUtil(self, v, visited, recStack): 
  
        # Mark current node as visited and  
        # adds to recursion stack 
        visited[v] = True
        recStack[v] = True
  
        # Recur for all neighbours 
        # if any neighbour is visited and in  
        # recStack then graph is cyclic 
        copyV = copy.copy(self.graph[v]);
        for neighbour in copyV: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                self.graph[v].remove(neighbour)               
                #return True
    
        # The node needs to be poped from  
        # recursion stack before function ends 
        recStack[v] = False
        return False
  
    # Returns true if graph is cyclic else false 
    def isCyclic(self): 
        visited = [False] * self.V 
        recStack = [False] * self.V 
        for node in range(self.V): 
            if visited[node] == False: 
                #if self.isCyclicUtil(node,visited,recStack) == True: 
                    #return True
                self.isCyclicUtil(node,visited,recStack)
        return False
                
    def getAdjacencyMatrix(self):
        AM = np.zeros((self.V, self.V))
        for g in self.graph:
            for i in self.graph[g]:
                AM[g, i] = 1
        
        return AM
#g = Graph(0, 0, 0, 0)

def main(n):

    g = Graph(n, 0, 1, 2)
    g.generateStruct(g)
    g.graphUnuque()
    g.isCyclic()

    AM = g.getAdjacencyMatrix()

    # search ending nods without connections
    u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[np.array(u) != 1]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
        else:
            AM[i, 1] = 1

    print(emptyRows)
    print(AM)
    io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})
    

    return AM

    
main(25)




"""if __name__ == '__main__':
    x = int(sys.argv[1])
    
    g = Graph(x, 0, 1, 3)

    main()
    g = Graph(x, 0, 1, 3)
    g.generateStruct()
    g.graphUnuque()
    g.isCyclic()

    AM = g.getAdjacencyMatrix()
    
    print(AM)"""



