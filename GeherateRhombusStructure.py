import numpy as np 
import scipy as sc 
import collections
import copy
import sys
from scipy import io

from scipy.stats import norm
import matplotlib.pyplot as plt

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
            self.addEdge(self.start, self.Layers[0][i])

        for i in np.arange(len(self.Layers[self.L - 1])):
            self.addEdge(self.end, self.Layers[self.L - 1][i])

        for i in np.arange(0, self.L - 1):
            curLayerNumElems = len(self.Layers[i])
            nextLayerNumElems = len(self.Layers[i + 1])

            if (curLayerNumElems < nextLayerNumElems):
                countOfGroups = min(curLayerNumElems, nextLayerNumElems)
                ind = np.array(self.Layers[i + 1]).reshape(-1, 1)
                for k in np.arange(ind.shape[0]):

                ind = np.random.choice(range(ind.shape[0]), size=len(ind), replace=False)
                groups = np.array_split(ind, countOfGroups)

                for j in np.arange(len(groups)):
                    self.graph[self.Layers[i][j]] = list(groups[j])

            if (curLayerNumElems > nextLayerNumElems):
                countOfGroups = nextLayerNumElems
                ind = np.array(self.Layers[i]).reshape(-1, 1)
                ind = np.random.choice(range(ind.shape[0]), size=len(ind), replace=False)
                groups = np.array_split(ind, countOfGroups)
                for j in np.arange(len(groups)):
                    for k in groups[j]:
                        self.addEdge([groups[j][k]], self.Layers[i][j])












    def addEdge(self, u, v): 
        self.graph[u].append(v) 
    
    def addElemInLayer(self, u, v): 
        self.Layers[u].append(v) 
    
    def graphUnuque(self):
        for i in self.graph:
            self.graph[i] = (list(set(self.graph[i])))

    


g = Graph(16, 3, 0, 1, 2)
g.calcLayers()
g.generateStruct()

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

