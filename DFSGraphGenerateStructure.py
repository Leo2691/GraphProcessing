import numpy as np
import collections

class Graph():
    def __init__(self, vertices, start, end):
        self.graph = collections.defaultdict(list) 
        self.V = vertices 
        self.start = start
        self.end = end

    def addEdge(self, u, v): 
        self.graph[u].append(v) 


    def generateStruct(self):
        visited = [0] * self.V 
        recStack = [0] * self.V 

        for i in np.arange(1, 100):
            # count connections with input
            countCon = np.random.randint(1, 4)
            # ids connections with input
            ids = np.unique([np.random.randint(3, self.V+1) for i in np.arange(countCon)]) 

            for j in ids:
                 # add conns with input 
                g.addEdge(start, j)
                if visited[j] == 0:
                    self.genCon(j, visited, recStack)

    def genCon(self, v, visited, recStack):
        visited[v] = 1
        recStack[v] = 1
        



        return 1
                


g = Graph(8, 1, 2)

g.generateStruct()
