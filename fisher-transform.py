import numpy as np

def fisher_transform(array):
    minimum = np.min(array)
    maximum = np.max(array)
    y = (array - minimum) / (maximum - minimum)
    y = np.clip(2*y - 1, -0.999, 0.999)
    return np.log((1+y) / (1-y))
