import numpy as np

def RandomBinaryNetwork(size):
    return np.random.randint(0, 2, (size, size))