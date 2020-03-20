import numpy as np
import collections
import copy
import sys
from scipy import io, sparse, stats
#import networkx as nx
import math
import glob
from typing import NamedTuple
import shutil
import os

from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class Graph():
    """
    Class Graph
    """

    def __init__(self, vertices, layers, start, end):
        self.graph = collections.defaultdict(list)  # list of lists for holding nodes
        self.Layers = collections.defaultdict(list)  # list of lists for holding elements of each layers
        self.L = layers  # count of layers
        self.V = int(vertices)  # count of nodes
        self.start = start  # start node
        self.end = end  # end node
        self.removedNodes = list()  # removed nodes

    def CalcRhomboidLayers(self):
        """
        The function generates rhomboid structures.
        The function is based on a normal distribution geometry.
        The number of elements in the layer is estimated as the Gausian value of various values x.
        Tasks:
            1. Estimation the number of elements in each layer.
            2. Random selection of elements in each layer.

        : param self.V: count of nodes
        : param self.L: count of layers

        :return:
        self.Layers: list of lists for holding elements of each layers
        """

        allElements = np.arange(self.V)  # count of nodes
        percentOfL = np.zeros(self.L)  # array for percents of the elements in each layer
        countOfL = np.zeros(self.L, dtype=int)  # array for number of elements in each layer
        countLocked = len(self.start) + len(self.end)

        mn = (1 + self.L - 2) / 2  # (min(x) + max(x)) / 2 # mathematical expectation of a normal distribution
        sgm = self.Sigma(self.L - 2)  # standard deviation

        bollmask = [True] * len(allElements)  # mask for  array of elements
        for i in self.start:
            bollmask[i] = False  # Input is not selectable
            self.addElemInLayer(0, i)
        for i in self.end:
            bollmask[i] = False  # Output is not selectable
            self.addElemInLayer(self.L - 1, i)

        #First and last layers is inputs and output

        # Layer cycle
        for i in np.arange(1, self.L - 1, 1):

            percentOfL[i] = round(norm.pdf(i, mn, sgm) * 100)  # percents of the elements in current layer
            countOfL[i] = int(percentOfL[i] * (self.V - countLocked) / 100)  # number of the elements in current layer
            ind = np.random.choice(allElements[bollmask], replace=False,
                                   size=countOfL[i])  # random selection of elements indexes in the current layer

            # Adding Elements to a Layer
            for j in ind:
                self.addElemInLayer(i, j)
                bollmask[j] = False

        # not all elements will use in in the model
        # we will take [80%, 100%] of elements randomly
        for i in np.arange(1, self.L - 1, 1):
            percentOfDel = np.random.choice(np.arange(0, 20, dtype=int)) / 100  # percents of removed elemets
            countOfDel = int(len(self.Layers[i]) * percentOfDel)  # number of removed elemets
            selection = np.random.choice((self.Layers[i]), replace=False, size=countOfDel)  # Delete elements
            if len(selection) > 0:
                self.Layers[i] = [elem for elem in self.Layers[i] if elem not in selection]

        # print(np.sum(percentOfL))
        # print(np.sum(countOfL))

        # Vizualuzation
        #x = [i for i in np.arange(start = 0, stop = self.L)]
        #y = norm.pdf(x, mn, sgm) * 100
        #plt.plot(x, y)
        #plt.show()
        #print(x)

    def Sigma(self, numLayers):
        """
        Function is for calculating standard deviation
        :param numLayers: number of layers in model
        :return: sgm: standard deviation
        """

        x = numLayers
        sgm = (x / 7 - 3 / 7 + 0.9 / (2.5 - 0.9)) * (2.5 - 0.9)
        return sgm

    def GenerateStruct(self):
        """
        Network graph generation. Matching connections between layers and elements in them.
        :param self.Layers: list of lists of elements in each layer
        :param self.start:  start node
        :param self.end:    end node

        :return:
        self.graph: list of lists for holding nodes
        """

        # layer cycle
        for i in np.arange(0, self.L-1):
            curLayerNumElems = len(self.Layers[i])  # num of element in current layer
            nextLayerNumElems = len(self.Layers[i + 1])  # num of element in next layer

            # if num of elems in current layer less than in next
            if (curLayerNumElems < nextLayerNumElems):
                countOfGroups = curLayerNumElems
                ind = np.array(self.Layers[i + 1])  # .reshape(-1, 1)
                groups = np.array_split(ind, countOfGroups)
                for j in np.arange(len(groups)):
                    self.graph[self.Layers[i][j]] = list(groups[j])

                    # additional cross connections
                    countCrossCon = int(math.ceil(
                        (len(groups[j]) / 100 * 30)))  # 30 - 30% of group elements - count of cross connections
                    ix = ~np.isin(ind, groups[j])
                    try:
                        crossCon = np.random.choice(ind[ix], countCrossCon, replace=False)
                        self.graph[self.Layers[i][j]] = self.graph[self.Layers[i][j]] + list(crossCon)
                    except:
                        print("layer without cross Connections")

            # if num of elems in current layer more or equal in next
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
                    try:
                        crossCon = np.random.choice(ind[ix], countCrossCon, replace=False)
                        self.addEdge(crossCon, self.Layers[i + 1][j])
                    except:
                        print("layer without cross Connections")

    def isCyclicUtil(self, v, visited, recStack):
        """
        The function of finding and removing loops in a graph
        :param v:
        :param visited:
        :param recStack:
        :return:
        """

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
        visited = [False] * (self.V)
        recStack = [False] * (self.V)
        for node in range(self.V):
            if visited[node] == False:
                # if self.isCyclicUtil(node,visited,recStack) == True:
                # return True
                self.isCyclicUtil(node, visited, recStack)
        return False

    def getAdjacencyMatrix(self):
        """
        Generating an adjacency matrix from a graph structure
        :return: Adjacency Matrix
        """
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


def checkAMOnCyclic(AM, GenAM, minWeigOfCon, start, end):
    """
    Checking the adjacency matrix for correctness.
    Loop removal.
    Top Level Collision Resolution

    :param AM:              Adjacency Matrix
    :param GenAM:           Smooth Genetic Matrix
    :param minWeigOfCon:    Minimum connection weight in a smooth matrix
    :param start:           start node
    :param end:             end node

    :return:
    AM, GenAM               Adjacency and mooth Genetic Matrices without loops and collisions
    """
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

    # the output does not refer to anything
    for i in end:
        AM[i, :] = 0
        GenAM[i, :] = 0
    # nothing comes to input
    for i in start:
        AM[:, i] = 0
        GenAM[:, i] = 0

    # lock angle where located inputs and outputs. we remove connnects input->output
    collsInOut = np.union1d(start, end)
    for i in collsInOut:
        for j in collsInOut:
            if(AM[i,j] != 0):
                AM[i,j] = 0
                GenAM[i, j] = 0

    #we define columns of inputs and outputs, to which the least references are made
    congestionInputs = np.array([[sum(AM[:, i]), i] for i in start]).reshape(-1,2)
    minColInput = congestionInputs[np.argmin(congestionInputs[:, 0]), 1]
    congestionOutputs = np.array([[sum(AM[:, i]), i] for i in end]).reshape(-1, 2)
    minColOutput = congestionOutputs[np.argmin(congestionOutputs[:, 0]), 1]

    # search ending nods without connections
    u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[~np.isin(np.array(u), end)]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
            GenAM[:, i] = 0
        else:
            AM[i, minColOutput] = 1
            GenAM[i, minColOutput] = minWeigOfCon

    # search nods without inputs
    u = np.array(np.where(AM.sum(axis=0) == 0))
    emptyColls = np.array(u)[~np.isin(np.array(u), start)]
    for i in emptyColls:
        if (AM[i, :].sum(axis=0) == 0):
            AM[i, :] = 0
            GenAM[i, :] = 0
        else:
            AM[minColInput, i] = 1
            GenAM[minColInput, i] = minWeigOfCon

    for i in np.arange(AM.shape[0]):
        for j in np.arange(AM.shape[1]):
            if i == j:
                AM[i, j] = 0
                GenAM[i, j] = 0

    n = sum(sum(AM))
    g1 = Graph(AM.shape[0], 1, start, end)
    for i in np.arange(AM.shape[0]):
        for j in np.arange(AM.shape[1]):
            if (AM[i, j] != 0):
                g1.addEdge([i], j)
    g1.isCyclic()
    AM = g1.getAdjacencyMatrix()

    ## del cyclic connections in GenAM
    for i in np.arange(len(g1.removedNodes)):
        x = g1.removedNodes[i][0]
        y = g1.removedNodes[i][1]
        GenAM[x, y] = 0

    # search ending nods without connections
    u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[~np.isin(np.array(u), end)]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
            GenAM[:, i] = 0
        else:
            AM[i, minColOutput] = 1
            GenAM[i, minColOutput] = minWeigOfCon

    # search nods without inputs
    u = np.array(np.where(AM.sum(axis=0) == 0))
    emptyColls = np.array(u)[~np.isin(np.array(u), start)]
    for i in emptyColls:
        if (AM[i, :].sum(axis=0) == 0):
            AM[i, :] = 0
            GenAM[i, :] = 0
        else:
            AM[minColInput, i] = 1
            GenAM[minColInput, i] = minWeigOfCon

    AM = np.array(AM, dtype=int)

    # comparision AM and GenAM
    m = np.where(AM > 0)  # indexis of nonzeros
    A = GenAM.astype(bool).astype(int)  # GenAM to [0,1]
    s1 = sum(AM[m[0], m[1]])  # sum of all nonzeros in AM with mask m
    s2 = sum(A[m[0], m[1]])  # sum of all nonzeros in A (GenAM in the past) with mask m

    return [AM, GenAM]

def checkAMOnCyclicInMatlab(AM_, GenAM = [], size = 0, minWeigOfCon = 0, start_node = 0, end_node = 0):
    """
    Checking the adjacency matrix for correctness.
    Loop removal.
    Top Level Collision Resolution

    :param AM:              Adjacency Matrix
    :param GenAM:           Smooth Genetic Matrix
    :param start:           start node
    :param end:             end node

    :return:
    AM, GenAM               Adjacency and mooth Genetic Matrices without loops and collisions
    """

    start = [int(i) for i in start_node]
    end = [int(i) for i in end_node]

    AM = np.array(AM_, dtype=int).reshape(int(size[0]), int(size[1]))

    if len(GenAM) == 0:
        GenAM_ = np.zeros((int(size[0]), int(size[1])))
    else:
        GenAM_ = np.array(GenAM, dtype=int).reshape(int(size[0]), int(size[1]))

    [AM_clear, GenAM_clear] = checkAMOnCyclic(AM, GenAM_, minWeigOfCon, start, end)

    if len(GenAM) == 0:
        return AM_clear
    else:
        return [AM_clear, GenAM_clear]
    #io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})


#checkAMOnCyclicInMatlab(1, 1, 1, [0], [1])

def main(num_elemetns, num_layers, start_node, end_node):
    #start_node = [0,1]
    #end_node = [2,3]
    start_node = [int(i) for i in start_node]
    end_node = [int(i) for i in end_node]
    g = Graph(num_elemetns, num_layers, start_node, end_node)
    g.CalcRhomboidLayers()
    g.GenerateStruct()

    AM = g.getAdjacencyMatrix()
    #G = nx.from_numpy_matrix(np.array(AM))
    #nx.draw(G, with_labels=True)
    #plt.show()

    #AM = checkAdjacencyMatrixOnCyclic(AM, AM, 0.2)

    #print(sum(sum(AM)))

    # search ending nods without connections
    """u = np.array(np.where(AM.sum(axis=1) == 0))
    emptyRows = np.array(u)[np.array(u) != 1]
    for i in emptyRows:
        if (AM[:, i].sum(axis=0) == 0):
            AM[:, i] = 0
        else:
            AM[i, 1] = 1"""

    #plt.imshow(AM)
    #ax = plt.gca()
    #ax.grid(color='w', linestyle='-', linewidth=2)
    #plt.show()

    # print(emptyRows)
    # print(AM)
    #io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})

    return AM

#checkAMOnCyclicInMatlab(1, [5,5], [0], [1])
#main(10, 4, [0.0, 1.0], [2.0, 3.0])

AM = np.array([[0, 0, 1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 0, 0, 1],
                   [0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   ])

AM_ = np.array([[0, 0, 1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 0, 0, 1],
                   [0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   ], dtype=float)

#[A, B] = checkAMOnCyclicInMatlab(AM, AM_, AM.shape, 0.0, [0], [1,2])
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

def test():
    A = np.random.randint(10, size=(10,10))
    B = np.random.randint(10, size=(10, 10))

    return [A, B]
