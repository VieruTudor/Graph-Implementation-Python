import random
import copy
import heapq


def nextPermutation(L):
    n = len(L)
    # Step 1: find rightmost position i such that L[i] < L[i+1]
    i = n - 2
    while i >= 0 and L[i] >= L[i + 1]:
        i -= 1

    if i == -1:
        return False

    j = i + 1
    while j < n and L[j] > L[i]:
        j += 1
    j -= 1
    L[i], L[j] = L[j], L[i]
    left = i + 1
    right = n - 1
    while left < right:
        L[left], L[right] = L[right], L[left]
        left += 1
        right -= 1
    return True


class Graph:
    def __init__(self, verticesNumber):
        self._inboundEdges = {}
        self._outboundEdges = {}
        self._verticesNumber = verticesNumber
        self._edgesNumber = 0
        self.tm = [0] * verticesNumber
        self.tsm = [0] * verticesNumber
        self.Tm = [999999999] * verticesNumber
        self.Tsm = [999999999] * verticesNumber
        self.duration = [0] * verticesNumber
        self.timeSet = False
        for vertex in range(verticesNumber):
            self._inboundEdges[vertex] = []
            self._outboundEdges[vertex] = []

    # region Getters

    def getVerticesNumber(self):
        return self._verticesNumber

    def getEdgesNumber(self):
        return self._edgesNumber

    def getInDegree(self, vertex):
        return len(self._inboundEdges[vertex])

    def getOutDegree(self, vertex):
        return len(self._outboundEdges[vertex])

    def getCost(self, start, end):
        if self.isEdge(start, end):
            costs = [vertex[1] for vertex in self._outboundEdges[start] if vertex[0] == end]
            cost = costs[0]
        else:
            raise Exception("Edge does not exist")

        return cost

    def getEdges(self):
        edges = []
        for vertex in self._outboundEdges:
            for edge in self._outboundEdges[vertex]:
                edges.append((vertex, edge[0]))
        return edges

    # endregion

    # region Setters
    def setEdgesNumber(self, edges):
        self._edgesNumber = edges

    def setCost(self, start, end, newCost):
        self.removeEdge(start, end)
        self.addEdge(start, end, newCost)

    # endregion

    # region Parsers
    def parseVertices(self):
        return self._outboundEdges.keys()

    def parseVertexOutbound(self, vertex):
        return self._outboundEdges[vertex]

    def parseVertexInbound(self, vertex):
        return self._inboundEdges[vertex]

    # endregion

    # region Graph Operations
    def addVertex(self, vertex):
        if vertex in self._inboundEdges.keys():
            raise Exception("Vertex already exists !")
        self._inboundEdges[vertex] = []
        self._outboundEdges[vertex] = []
        self._verticesNumber += 1

    def removeVertex(self, deletedVertex):
        if deletedVertex not in self._inboundEdges.keys():
            raise Exception("Vertex does not exist")
        self._outboundEdges.pop(deletedVertex)
        self._inboundEdges.pop(deletedVertex)
        for vertex in self._inboundEdges:
            self._inboundEdges[vertex] = [edge for edge in self._inboundEdges[vertex] if edge[0] != deletedVertex]
        for vertex in self._outboundEdges:
            self._outboundEdges[vertex] = [edge for edge in self._outboundEdges[vertex] if edge[0] != deletedVertex]
        self._verticesNumber -= 1

    def addEdge(self, start, end, cost):
        if not self.isEdge(start, end):
            self._outboundEdges[start].append([end, cost])
            self._inboundEdges[end].append([start, cost])
            self._edgesNumber += 1
        else:
            raise Exception("Edge already exists")



    def isEdge(self, start, end):
        return end in [vertex[0] for vertex in self._outboundEdges[start]]

    def removeEdge(self, start, end):
        if self.isEdge(start, end):
            self._outboundEdges[start] = [vertex for vertex in self._outboundEdges[start] if vertex[0] != end]
            self._inboundEdges[end] = [vertex for vertex in self._inboundEdges[end] if vertex[0] != start]
            self._edgesNumber -= 1
        else:
            raise Exception("Edge does not exist")

    # endregion

    def printGraph(self):
        print("Outbound edges")
        print(self._outboundEdges)
        print("Inbound edges")
        print(self._inboundEdges)
        print('\n')

    def copyGraph(self):
        return copy.deepcopy(self)

    def accessible(self, vertex):
        """Returns the set of vertices of the graph g that are accessible
        from the vertex s"""
        acc = set()
        acc.add(vertex)
        list = [vertex]
        while len(list) > 0:
            x = list[0]
            list = list[1:]
            for y in self.parseVertexOutbound(x):
                if y not in acc:
                    acc.add(y)
                    list.append(y)
        return acc

    def reverseBFS_shortestPath(self, startVertex, endVertex):
        queue = []
        previous = {}
        distances = {}
        visited = set()
        queue.append(endVertex)
        visited.add(endVertex)
        distances[endVertex] = 0
        while len(queue) != 0:
            x = queue.pop(0)
            for y in self.parseVertexInbound(x):
                if y[0] not in visited:
                    queue.append(y[0])
                    visited.add(y[0])
                    distances[y[0]] = distances[x] + 1
                    previous[y[0]] = x

        path = []

        dist = distances[startVertex]
        key = startVertex
        while dist > 0:
            path.append(key)
            key = previous[key]
            dist -= 1

        path.append(key)
        return path, distances[startVertex]

    def dijkstra(self, startVertex):
        previous = {}
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (0, startVertex))
        d = {startVertex: 0}
        visited = set()
        visited.add(startVertex)
        while len(queue) > 0:
            print("Queue:", queue)
            x = heapq.heappop(queue)[1]
            print("Vertex: ", str(x))
            for y in self.parseVertexOutbound(x):
                if y[0] not in visited or d[y[0]] > d[x] + self.getCost(x, y[0]):
                    d[y[0]] = d[x] + self.getCost(x, y[0])
                    visited.add(y[0])
                    heapq.heappush(queue, (d[y[0]], y[0]))
                    previous[y[0]] = x
            print("Distances: ", d)
            print("Previous: ", previous)
        return d

    def BellmanFord(self, startVertex, endVertex):
        distance = [{startVertex: 0}]
        prevVertices = {}
        for i in range(1, self._verticesNumber):
            previous = distance[i - 1]
            current = {}
            for y in previous:
                for x in self.parseVertexOutbound(y):
                    if x[0] not in current or current[x[0]] > previous[y] + self.getCost(y, x[0]):
                        current[x[0]] = previous[y] + self.getCost(y, x[0])
                        prevVertices[x[0]] = y
            distance.append(current)
        print(distance)

        # getting the minimum Cost for the assigned vertex
        minCost = float("Inf")
        length = 0
        for i in range(len(distance)):
            for key in distance[i].keys():
                if key == endVertex:
                    if distance[i][key] < minCost:
                        minCost = distance[i][key]
                        length = i

        # computing the path
        path = []
        key = endVertex
        while length > 0:
            path.append(key)
            key = prevVertices[key]
            length -= 1

        path.append(key)

        return minCost, list(reversed(path))

    def topologicalSort(self):
        sortedList = []
        queue = []
        count = {}
        for vertex in self.parseVertices():
            count[vertex] = self.getInDegree(vertex)
            if count[vertex] == 0:
                queue.append(vertex)
        while len(queue) != 0:
            vertex = queue.pop(0)
            sortedList.append(vertex)
            for y in self.parseVertexOutbound(vertex):
                count[y[0]] = count[y[0]] - 1
                if count[y[0]] == 0:
                    queue.append(y[0])
        if len(sortedList) < self._verticesNumber:
            raise Exception("NOT A DAG")
        return sortedList

    def isCycle(self, vertices):
        cycleVertices = copy.copy(vertices)
        cycleVertices.append(cycleVertices[0])
        for index in range(1, len(cycleVertices)):
            if not self.isEdge(cycleVertices[index - 1], cycleVertices[index]):
                return False
        return True

    def computeCost(self, vertices):
        costVertices = copy.deepcopy(vertices)
        costVertices.append(vertices[0])
        cost = 0
        for index in range(1, len(costVertices)):
            cost += self.getCost(costVertices[index - 1], costVertices[index])
        return cost

    def minimumHamiltonianCost(self):
        minCost = float('inf')
        vertices = list(self.parseVertices())
        hamiltonianCycle = []
        while True:
            if self.isCycle(vertices):
                cost = self.computeCost(vertices)
                if cost < minCost:
                    minCost = cost
                    hamiltonianCycle = copy.deepcopy(vertices)
            if not nextPermutation(vertices):
                break
        return hamiltonianCycle, minCost

    def setTimes(self):
        sortedGraph = self.topologicalSort()
        if not self.timeSet:
            for x in sortedGraph:
                for y in self.parseVertexInbound(x):
                    self.tm[x] = max(self.tm[x], self.tsm[y[0]])
                self.tsm[x] = self.tm[x] + self.duration[x]
            self.Tm[self.getVerticesNumber() - 1] = self.Tsm[self.getVerticesNumber() - 1] = self.tsm[
                self.getVerticesNumber() - 1]
            for x in reversed(sortedGraph):
                for y in self.parseVertexOutbound(x):
                    self.Tsm[x] = min(self.Tsm[x], self.Tm[y[0]])
                self.Tm[x] = self.Tsm[x] - self.duration[x]
            self.timeSet = True

    def getDuration(self):
        self.setTimes()
        print("Total time: ", self.tm[self.getVerticesNumber() - 1])
        for i in range(1, self.getVerticesNumber() - 1):
            print([self.tm[i], i - 1, self.Tm[i]])

    def getCritical(self):
        self.setTimes()
        arr = []
        for i in range(1, self.getVerticesNumber() - 1):
            if self.tm[i] == self.Tm[i]:
                arr.append(i - 1)
        print("Critical activities: ", arr)


def initRandomGraph(constructor, verticesNumber, edgesNumber):
    g = constructor(verticesNumber)
    addedEdges = 0
    while addedEdges < edgesNumber:
        start = random.randrange(0, verticesNumber)
        end = random.randrange(0, verticesNumber)
        cost = random.randrange(0, 10)
        if not g.isEdge(start, end):
            g.addEdge(start, end, cost)
            addedEdges += 1
    return g


def readFromFile(file):
    with open(file, 'r') as reader:
        firstLine = reader.readline().strip()
        firstLine = firstLine.split(" ")
        vertices = int(firstLine[0])
        edges = int(firstLine[1])
        g = Graph(vertices)
        lines = reader.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(" ")
            g.addEdge(int(line[0]), int(line[1]), int(line[2]))
            g.setEdgesNumber(edges)
    return g


def readActivities():
    f = open('activities.txt', 'r')
    line = f.readline()
    line = line.strip()
    nrActivities = line
    G = Graph(int(nrActivities) + 2)
    line = f.readline()
    line = line.strip()
    line = line.split()
    remaining = []
    for i in range(1, int(nrActivities) + 1):
        remaining.append(i)
    for i in range(int(nrActivities)):
        G.duration[i + 1] = int(line[i])
    for i in range(int(nrActivities)):
        line = f.readline()
        line = line.strip()
        line = line.split()
        for j in range(len(line)):
            G.addEdge(int(line[j]) + 1, i + 1, 0)
            if (int(line[j]) + 1) in remaining:
                remaining.remove(int(line[j]) + 1)
    for el in remaining:
        G.addEdge(el, int(nrActivities) + 1, 0)
    line = f.readline()
    if line != '':
        line = int(line)
        for i in range(line):
            line = f.readline()
            line = line.strip()
            line = line.split()
            G.setCost(int(line[0]) + 1, int(line[1]) + 1, int(line[2]))
    return G


def writeToFile(graph, file):
    with open(file, 'w') as writer:
        firstLine = str(graph.getVerticesNumber()) + " " + str(graph.getEdgesNumber())
        writer.write(firstLine + '\n')
        for vertex in graph.parseVertices():
            for edge in graph.parseVertexOutbound(vertex):
                line = str(vertex) + " " + str(edge[0]) + " " + str(edge[1])
                writer.write(line + '\n')


def testGraphFunctions():
    graph = readFromFile("test.txt")
    writeToFile(graph, "out.txt")
    print("1. Get the number of vertices in the graph\n"
          "2. Iterate(display) the set of vertices\n"
          "3. Check if there exists an edge between 2 vertices\n"
          "4. Display inbound and outbound degree for a vertex\n"
          "5. Display outbound edges of a vertex\n"
          "6. Display inbound edges of a vertex\n"
          "7. Get the cost of an edge\n"
          "8. Change the cost of an edge\n"
          "9. Add a vertex\n"
          "10. Remove a vertex\n"
          "11. Add an edge\n"
          "12. Remove an edge\n"
          "13. Print the graph")
    while True:
        command = input(">>>")
        if command == "1":
            print(graph.getVerticesNumber())
        if command == "2":
            print(graph.parseVertices())
        if command == "3":
            start = int(input("Input start vertex: "))
            end = int(input("Input end vertex: "))
            print(graph.isEdge(start, end))
        if command == "4":
            vertex = int(input("Vertex number: "))
            print("In degree is: " + str(graph.getInDegree(vertex)))
            print("Out degree is: " + str(graph.getOutDegree(vertex)))
        if command == "5":
            vertex = int(input("Vertex number: "))
            print(graph.parseVertexOutbound(vertex))
        if command == "6":
            vertex = int(input("Vertex number: "))
            print(graph.parseVertexInbound(vertex))
        if command == "7":
            start = int(input("Input start vertex: "))
            end = int(input("Input end vertex: "))
            print(graph.getCost(start, end))
        if command == "8":
            start = int(input("Input start vertex: "))
            end = int(input("Input end vertex: "))
            newCost = int(input("Input new cost: "))
            graph.setCost(start, end, newCost)
        if command == "9":
            newVertex = int(input("Vertex to be added: "))
            graph.addVertex(newVertex)
        if command == "10":
            deletedVertex = int(input("Vertex to be removed: "))
            graph.removeVertex(deletedVertex)
        if command == "11":
            start = int(input("Input start vertex: "))
            end = int(input("Input end vertex: "))
            cost = int(input("Input cost: "))
            graph.addEdge(start, end, cost)

        if command == "12":
            start = int(input("Input start vertex: "))
            end = int(input("Input end vertex: "))
            graph.removeEdge(start, end)

        if command == "13":
            graph.printGraph()

        if command == "14":
            verticesNumber = int(input("Number of vertices: "))
            edgesNumber = int(input("Number of edges: "))
            graph = initRandomGraph(Graph, verticesNumber, edgesNumber)


graph = readFromFile("test.txt")
print(graph.dijkstra(0))


# region Seminary
# def getTree(graph, startVertex):
#     tree = {}
#     root = startVertex
#     visited = set()
#     queue = []
#     tree[root] = []
#     visited.add(startVertex)
#     queue.append(startVertex)
#     while len(queue) != 0:
#         source = queue.pop(0)
#         for neighbour in graph.parseVertexOutbound(source):
#             if neighbour[0] not in visited:
#                 tree[neighbour[0]] = []
#                 visited.add(neighbour[0])
#                 queue.append(neighbour[0])
#                 tree[source].append(neighbour[0])
#     return tree, root
#
#
# def printTree(tree, root, Tab):
#     print(Tab + str(root))
#     for children in tree[root]:
#         printTree(tree, children, Tab + "\t")
#
#
# g = initRandomGraph(Graph, 4, 7)
# g.printGraph()
# tree, root = getTree(g, 2)
#
# printTree(tree, root, "")
# endregion
