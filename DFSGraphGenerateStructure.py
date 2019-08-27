import numpy as np
import collections

class Graph():
    def __init__(self, vertices):
        self.graph = collections.defaultdict(list) 
        self.V = vertices 


    def addEdge(self, u, v): 
            self.graph[u].append(v) 


    def generateStruct(self, start, end):
        visited = [False] * self.V 
        recStack = [False] * self.V 

        for i in np.arange(1, self.V ):
            countCon = np.random.randint(1, 4)
            
            for j in np.arange(countCon):
                a = j


g = Graph(6)

g.generateStruct(1, 2)
