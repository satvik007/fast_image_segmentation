import numpy as np


class UnionFind:

    def __init__(self, n):
        self.n = n
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)
        self.csize = np.ones(n, dtype=np.int32)
        self.min_edge = np.zeros(n)

    def find(self, u):
        v = u
        while u != self.parent[u]:
            u = self.parent[u]
        while v != self.parent[v]:
            t = self.parent[v]
            self.parent[v] = u
            v = t
        return u

    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u != v:
            if self.rank[u] < self.rank[v]:
                self.parent[u] = v
                self.csize[v] += self.csize[u]
                self.min_edge[v] = min(self.min_edge[v], self.min_edge[u])
            else:
                self.parent[v] = u
                self.csize[u] += self.csize[v]
                self.min_edge[u] = min(self.min_edge[u], self.min_edge[v])
                if self.rank[u] == self.rank[v]:
                    self.rank[u] += 1

    def is_same_set(self, u, v):
        return self.find(u) == self.find(v)