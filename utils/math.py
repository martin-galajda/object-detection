import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
