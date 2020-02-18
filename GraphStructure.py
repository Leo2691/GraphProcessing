import numpy as np
import collections
import copy
import sys
from scipy import io, sparse, stats
# import networkx as nx
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

        mn = (0 + self.L - 1) / 2  # (min(x) + max(x)) / 2 # mathematical expectation of a normal distribution
        sgm = self.Sigma(self.L - 1)  # standard deviation

        bollmask = [True] * len(allElements)  # mask for  array of elements
        for i in self.start:
            bollmask[i] = False # Input is not selectable
        for i in self.end:
            bollmask[i] = False  # Output is not selectable
        # Layer cycle
        for i in np.arange(self.L):

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
        for i in np.arange(self.L):
            percentOfDel = np.random.choice(np.arange(0, 20, dtype=int)) / 100  # percents of removed elemets
            countOfDel = int(len(self.Layers[i]) * percentOfDel)  # number of removed elemets
            selection = np.random.choice((self.Layers[i]), replace=False, size=countOfDel)  # Delete elements
            if len(selection) > 0:
                self.Layers[i] = [elem for elem in self.Layers[i] if elem not in selection]

        # print(np.sum(percentOfL))
        # print(np.sum(countOfL))

        # Vizualuzation
        # x = [i for i in np.arange(start = 0, stop = self.L)]
        # y = norm.pdf(x, mn, sgm) * 100
        # plt.plot(x, y)
        # plt.show()
        # print(x)

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
        # connection strat node with second layer
        for i in np.arange(len(self.Layers[self.start])):
            input = np.random.choice(self.start, size=1)
            self.addEdge([self.start], self.Layers[0][i])

        # connection end-1 layer with end node
        self.addEdge(self.Layers[self.L - 1], self.end)

        # layer cycle
        for i in np.arange(0, self.L - 1):
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


def checkAMOnCyclic(AM, GenAM, minWeigOfCon):
    """
    Checking the adjacency matrix for correctness.
    Loop removal.
    Top Level Collision Resolution

    :param AM:              Adjacency Matrix
    :param GenAM:           Smooth Genetic Matrix
    :param minWeigOfCon:    Minimum connection weight in a smooth matrix
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

    for i in np.arange(AM.shape[0]):
        for j in np.arange(AM.shape[1]):
            if i == j:
                AM[i, j] = 0
                GenAM[i, j] = 0

    n = sum(sum(AM))
    g1 = Graph(AM.shape[0], 1, 0, 1)
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

    # comparision AM and GenAM
    m = np.where(AM > 0)  # indexis of nonzeros
    A = GenAM.astype(bool).astype(int)  # GenAM to [0,1]
    s1 = sum(AM[m[0], m[1]])  # sum of all nonzeros in AM with mask m
    s2 = sum(A[m[0], m[1]])  # sum of all nonzeros in A (GenAM in the past) with mask m

    return [AM, GenAM]


def main(num_elemetns, num_layers):
    start_node = [0,1]
    end_node = [2]
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
    #plt.show()

    # print(emptyRows)
    # print(AM)
    io.savemat('AdjacencyMatrix.mat', mdict={'AM': AM})

    return AM

main(10, 3)


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
