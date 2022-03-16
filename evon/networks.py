import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def RandomBinaryNetwork(size):
    return np.random.randint(0, 2, (size, size))

def PlotNetwork(best):
    rows, cols = np.where(best == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr)
    plt.show()