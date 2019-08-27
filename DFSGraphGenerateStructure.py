import numpy as np
import collections

class Graph():
    def __init__(self, vertices):
        self.graph = collections.defaultdict(list) 
        self.V = vertices 


    def addEdge(self, u, v): 
            self.graph[u].append(v) 


    def generateStruct(self, start, end):
        visited = [0] * self.V 
        recStack = [0] * self.V 

        for i in np.arange(1, 100):
            countCon = np.random.randint(1, 4)
            ids = np.unique([np.random.randint(3, self.V+1) for i in np.arange(countCon)])
            
            for j in ids:
                if visited[j] == 0:
                    self.genCon(j, visited, recStack)

    def genCon(self, v, visited, recStack):
        visited[v] = 1
        recStack[v] = 1
        



        return 1
                


g = Graph(8)

g.generateStruct(1, 2)
