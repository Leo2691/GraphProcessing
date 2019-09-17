import numpy as np 
import scipy as sc 
import collections
import copy
import sys
from scipy import io

from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

class Graph():
    def __init__(self, vertices, layers, start, end, limitCon):
        self.graph = collections.defaultdict(list)
        self.Layers = collections.defaultdict(list)
        self.L = layers 
        self.V = vertices 
        self.start = start
        self.end = end
        self.limitCon = limitCon

    def calcLayers(self):

        allElements = np.arange(self.V)
        percentOfL = np.zeros(self.L)
        countOfL = np.zeros(self.L, dtype=int)
        
        mn = (0 + self.L - 1) / 2 # (min(x) + max(x)) / 2 
        sgm = self.Sigma(self.L - 1) 

        bollmask = [True] * len(allElements)
        bollmask[self.start] = False
        bollmask[self.end] = False
        for i in np.arange(self.L):

            percentOfL[i] = round(norm.pdf(i, mn, sgm) * 100)
            countOfL[i] = int(percentOfL[i] * (self.V - 2) / 100)
            ind = np.random.choice(allElements[bollmask], replace=False, size=countOfL[i])
            
            for j in ind:
                self.addElemInLayer(i, j)
                bollmask[j] = False

        #print(np.sum(percentOfL))
        #print(np.sum(countOfL))

        # Vizualuzation
        #x = [i for i in np.arange(start = 0, stop = self.L)] 
        # y = norm.pdf(x, mn, sgm) * 100
        #plt.plot(x, y) 
        #plt.show() 
        #print(x)


    def Sigma(self, numLayers): 
        x = numLayers;
        sgm = (x/7 - 3/7 + 0.9 / (2.5 - 0.9)) * (2.5 - 0.9) 
        return sgm

    def generateStruct(self):

        for i in np.arange(len(self.Layers[self.start])):
            self.addEdge([self.start], self.Layers[0][i])

        #for i in np.arange(len(self.Layers[self.L - 1])):
        self.addEdge(self.Layers[self.L - 1], self.end)

        for i in np.arange(0, self.L - 1):
            curLayerNumElems = len(self.Layers[i])
            nextLayerNumElems = len(self.Layers[i + 1])

            if (curLayerNumElems < nextLayerNumElems):
                countOfGroups = curLayerNumElems
                ind = np.array(self.Layers[i + 1])#.reshape(-1, 1)
                groups = np.array_split(ind, countOfGroups)
                for j in np.arange(len(groups)):
                    self.graph[self.Layers[i][j]] = list(groups[j])

            if (curLayerNumElems > nextLayerNumElems):
                countOfGroups = nextLayerNumElems
                ind = np.array(self.Layers[i])
                groups = np.array_split(ind, countOfGroups)
                for j in np.arange(len(groups)):
                    self.addEdge(groups[j], self.Layers[i + 1][j])

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
                # return True

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
                # if self.isCyclicUtil(node,visited,recStack) == True:
                # return True
                self.isCyclicUtil(node, visited, recStack)
        return False

    def getAdjacencyMatrix(self):
        AM = np.zeros((self.V, self.V))
        for g in self.graph:
            for i in self.graph[g]:
                AM[g, i] = 1

        return AM

    def addEdge(self, u, v):
        u = np.array(u)
        for i in np.arange(len(u)):
            self.graph[u[i]].append(v)
    
    def addElemInLayer(self, u, v): 
        self.Layers[u].append(v) 
    
    def graphUnuque(self):
        for i in self.graph:
            self.graph[i] = (list(set(self.graph[i])))

    


""".imshg = Graph(16, 3, 0, 1, 2)
g.calcLayers()
g.generateStruct()
AM = g.getAdjacencyMatrix()
pltow(AM)
plt.show()

t = 1"""


def main(n, layers, limCon):
    g = Graph(n, layers, 0, 1, 2)
    g.calcLayers()
    g.generateStruct()

    AM = g.getAdjacencyMatrix()

    # search ending nods without connections
    """u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[np.array(u) != 1]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
        else:
            AM[i, 1] = 1"""

    #plt.imshow(AM)
    #plt.show()

    # print(emptyRows)
    print(AM)
    io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})

    return AM

main(39, 4, 4)


"""x = [i for i in np.arange(start = 1, stop = 4)] 
mn = (min(x) + max(x)) / 2 
sgm = Sigma(max(x)) 

y = norm.pdf(x, mn, sgm) * 100
y1 = norm.pdf(1, mn, sgm) * 100
y2 = norm.pdf(2, mn, sgm) * 100
sum = 2 * y1 + y2
plt.plot(x, y) 
plt.show() 
print(x)"""

