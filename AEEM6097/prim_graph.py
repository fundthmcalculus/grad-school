import heapq

# import numba
import numpy as np


class VATGraph:
    def __init__(self, adj: np.ndarray):
        self.V: int = len(adj)
        self.adj = adj

    # Function to print MST using Prim's algorithm
    # @numba.njit
    def prim_mst(self):
        pq = []  # Priority queue to store vertices that are being processed
        # Find the column of the maximum value.
        max_adj = np.argmax(self.adj)
        src = max_adj // self.V
        src_key = np.max(self.adj)

        # Create a list for keys and initialize all keys as infinite (INF)
        key = [float('inf')] * self.V

        # To store the parent array which, in turn, stores MST
        parent = [-1] * self.V

        # To keep track of vertices included in MST
        in_mst = [False] * self.V

        # Insert the source itself into the priority queue and initialize its key as 0
        heapq.heappush(pq, (src_key, src))
        key[src] = src_key

        heap_seq = []

        # Loop until the priority queue becomes empty
        while pq:
            # The first vertex in the pair is the minimum key vertex
            # Extract it from the priority queue
            # The vertex label is stored in the second of the pair
            u = heapq.heappop(pq)[1]

            # Different key values for the same vertex may exist in the priority queue.
            # The one with the least key value is always processed first.
            # Therefore, ignore the rest.
            if in_mst[u]:
                continue

            in_mst[u] = True  # Include the vertex in MST
            heap_seq.append(u)

            # Iterate through all adjacent vertices of a vertex
            for v in range(self.V):
                if v == u:
                    continue
                weight = self.adj[u, v]
                # If v is not in MST and the weight of (u, v) is smaller than the current key of v
                if not in_mst[v] and key[v] > weight:
                    # Update the key of v
                    key[v] = weight
                    heapq.heappush(pq, (key[v], v))
                    parent[v] = u

        return heap_seq
