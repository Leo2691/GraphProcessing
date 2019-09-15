import numpy as np 
import scipy as sc 
import collections
import copy
import sys
from scipy import io

from scipy.stats import norm
import matplotlib.pyplot as plt 

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

    def Sigma(numLayers): 
        x = numLayers;
        sgm = (x/7 - 3/7 + 0.9 / (2.5 - 0.9)) * (2.5 - 0.9) 
        return sgm 

x = [i for i in np.arange(start = 1, stop = 4)] 
mn = (min(x) + max(x)) / 2 
sgm = Sigma(max(x)) 

y = norm.pdf(x, mn, sgm) * 100
y1 = norm.pdf(1, mn, sgm) * 100
y2 = norm.pdf(2, mn, sgm) * 100
sum = 2 * y1 + y2
plt.plot(x, y) 
plt.show() 
print(x)