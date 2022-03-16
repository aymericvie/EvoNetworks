import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def CountingOnes(adjacency):
    return adjacency.sum()

size = 10
adjacency_matrix = np.random.randint(0, 2, (size, size))
print(adjacency_matrix)

print(CountingOnes(adjacency_matrix))


# def show_graph_with_labels(adjacency_matrix):
rows, cols = np.where(adjacency_matrix == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr)
# plt.show()


np.random.seed(8)