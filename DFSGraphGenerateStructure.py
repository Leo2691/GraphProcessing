import numpy as np
import collections

class Graph():
    def __init__(self, vertices, start, end, limitCon):
        self.graph = collections.defaultdict(list) 
        self.V = vertices 
        self.start = start
        self.end = end
        self.limitCon = limitCon

    def addEdge(self, u, v): 
        self.graph[u].append(v) 


    def generateStruct(self):
        visited = np.array([0] * self.V)
        elementsMask = [True] * self.V 
        elementsMask[self.start] = False 

        for i in np.arange(1, 100):
            # count connections with input
            countCon = np.random.randint(1, self.limitCon + 1)
            # ids connections with input
            ids = np.unique([np.random.randint(self.limitCon, self.V+1) for i in np.arange(countCon)]) 

            for j in ids:
                 # add conns with input 
                g.addEdge(self.start, j)
                if visited[j] <= self.limitCon:
                    self.genCon(j, visited, elementsMask)

    def genCon(self, v, visited, elementsMask):
        visited[v] += 1 
        #elementsMask[v] = False

         # count connections with input
        countCon = np.random.randint(1, self.limitCon + 1)
        # list elements for chousen
        posiblElems = np.arange(self.V)[elementsMask]
        # ids connections with input
        ids = np.unique([np.random.choice(posiblElems) for i in np.arange(countCon)]) 
        



        return 1
                
t = [False] * 8
a = np.zeros(8)

t[1] = True
t[3] = True

b = a[t]


r = np.arange(8)

g = Graph(8, 0, 1, 3)

g.generateStruct()


