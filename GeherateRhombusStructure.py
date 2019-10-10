import numpy as np
import scipy as sc 
import collections
import copy
import sys
from scipy import io
#import networkx as nx
import math
import glob
from typing import NamedTuple

from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

class Graph():
    def __init__(self, vertices, layers, start, end, limitCon):
        self.graph = collections.defaultdict(list) # list of lists for holding nodes
        self.Layers = collections.defaultdict(list) #list of lists for holding elements of each layers
        self.L = layers #count of layers
        self.V = int(vertices)  #count of nides
        self.start = start # start node
        self.end = end #end node
        self.limitCon = limitCon #limit of connections
        self.removedNodes = list()

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

        # not all elements will use in in the model
        # we will take [60%, 100%] of elements randomly
        for i in np.arange(self.L):
            percentOfDel = np.random.choice(np.arange(0, 45, dtype=int)) / 100
            countOfDel = int(len(self.Layers[i]) * percentOfDel)
            selection = np.random.choice((self.Layers[i]), replace=False, size=countOfDel)
            if len(selection) > 0:
                self.Layers[i] = [elem for elem in self.Layers[i] if elem not in selection]

        # print(np.sum(percentOfL))
        #print(np.sum(countOfL))

        # Vizualuzation
        #x = [i for i in np.arange(start = 0, stop = self.L)] 
        # y = norm.pdf(x, mn, sgm) * 100
        #plt.plot(x, y) 
        #plt.show() 
        #print(x)

    def Sigma(self, numLayers): 
        x = numLayers
        sgm = (x/7 - 3/7 + 0.9 / (2.5 - 0.9)) * (2.5 - 0.9) 
        return sgm

    def generateStruct(self):

        #connection strat node with second layer
        for i in np.arange(len(self.Layers[self.start])):
            self.addEdge([self.start], self.Layers[0][i])

        # connection end-1 layer with end node
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

                    #additional cross connections
                    countCrossCon = int(math.ceil(
                        (len(groups[j]) / 100 * 30))) #30 - 30% of group elements - count of cross connections
                    ix = ~np.isin(ind, groups[j])
                    crossCon = np.random.choice(ind[ix], countCrossCon, replace=False)
                    self.graph[self.Layers[i][j]] = self.graph[self.Layers[i][j]] + list(crossCon)

            if (curLayerNumElems >= nextLayerNumElems):
                countOfGroups = nextLayerNumElems
                ind = np.array(self.Layers[i])
                groups = np.array_split(ind, countOfGroups)
                for j in np.arange(len(groups)):
                    self.addEdge(groups[j], self.Layers[i + 1][j])

                    # additional cross connections
                    countCrossCon = int(math.ceil(
                        (len(groups[j]) / 100 * 30)))  # 30 - 30% of group elements - count of cross connections
                    ix = ~np.isin(ind, groups[j])
                    crossCon = np.random.choice(ind[ix], countCrossCon, replace=False)
                    self.addEdge(crossCon, self.Layers[i + 1][j])


    def isCyclicUtil(self, v, visited, recStack):

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        copyV = copy.copy(self.graph[v])
        for neighbour in copyV:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                self.graph[v].remove(neighbour)
                self.removedNodes.append([v, neighbour])
                # return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * (self.V-1)
        recStack = [False] * (self.V-1)
        for node in range(self.V-1):
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

    
def checkAMOnCyclic(AM, GenAM, minWeigOfCon):

    """AM = np.array([[0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0],
          [1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0]])

    GenAM = np.array([[0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0]], dtype=float)"""


    # search ending nods without connections
    u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[np.array(u) != 1]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
            GenAM[:, i] = 0
        else:
            AM[i, 1] = 1
            GenAM[i, 1] = minWeigOfCon

    # search nods without inputs
    u = np.array(np.where(AM.sum(axis=0) == 0))
    emptyColls = np.array(u)[np.array(u) != 0]
    for i in emptyColls:
        if (AM[i, :].sum(axis=0) == 0):
            AM[i, :] = 0
            GenAM[i, :] = 0
        else:
            AM[0, i] = 1
            GenAM[0, i] = minWeigOfCon

    n = sum(sum(AM))
    g1 = Graph(n, 1, 0, 1, 2)
    for i in np.arange(AM.shape[0]):
        for j in np.arange(AM.shape[1]):
            if(AM[i, j] != 0):
                g1.addEdge([i], j)
    g1.isCyclic()
    AM = g1.getAdjacencyMatrix()

    ## del cyclic connections in GenAM
    for i in np.arange(len(g1.removedNodes)):
        x = g1.removedNodes[i][0]
        y = g1.removedNodes[i][1]
        GenAM[x, y] = 0

    return [AM, GenAM]

## class for kepping parent info
class Parent(NamedTuple):
    path: str
    GenAM: np.array = 0
    WeigGenAM: np.array = 0

def generateChildrens(path):
    filesParentsGenAM = [f for f in glob.glob(path + "\parents\**\GenAM*.mat", recursive=True)]

    dictParents = {}
    i = 0
    for f in filesParentsGenAM:
        GenAM = io.loadmat(f.replace('\\', '/', 1))
        WeigGenAM = io.loadmat(f.replace('GenAM', 'WeigGenAM', 1))

        par = Parent(f, GenAM['GenAM'], WeigGenAM['WeigGenAM'])
        x = {i: par}
        dictParents.update(x)
        i += 1

    children_path = path + '\children'
    n_parents = i
    n_child = 0.5 * n_parents # 50% from all parents is count of new children

    for i in np.arange(n_child):
        # path to the new kid
        pth = children_path + '/' + str(int(i+1))
        n = np.arange(2, 5, 1)
        count_parents = np.random.choice(n, 1)
        pr = np.arange(0, n_parents)
        parents = np.random.choice(pr, replace=False, size=count_parents)

        new_kid = dictParents[0].WeigGenAM * 0
        countOfConnections = 0

        for j in np.arange(count_parents):
            new_kid = new_kid + dictParents[parents[j]].WeigGenAM
            countOfConnections += np.count_nonzero(dictParents[parents[j]].WeigGenAM)

        new_kid = new_kid / count_parents
        countOfConnections = int(countOfConnections / count_parents)

        GenAM = new_kid / np.amax(new_kid)

        # max powerfull connections in matrix
        valuesCon = sorted(GenAM.ravel(), reverse=True)[0:countOfConnections]
        mask = np.array(np.isin(GenAM, valuesCon))
        AM = mask * 1

        #checking AM correction
        minWaightOfCon = min(valuesCon)

        [AM, GenAM] = checkAMOnCyclic(AM, GenAM, minWaightOfCon)






        io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})
        io.savemat('AdjacencyMatrix.mat', mdict={'GenAM': AM})

        print(path)

path = './Experiments/GeneticAlgorithmExp1/Generations/1'
generateChildrens(path)

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
    #G = nx.from_numpy_matrix(np.array(AM))
    #nx.draw(G, with_labels=True)
    #plt.show()

    #AM = checkAdjacencyMatrixOnCyclic(AM, AM, 0.2)

    print(sum(sum(AM)))

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
    # print(AM)
    io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})

    return AM

main(50, 4, 4)


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

