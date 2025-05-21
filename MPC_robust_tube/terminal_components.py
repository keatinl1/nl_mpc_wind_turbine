import numpy as np
import scipy.io as sio

mat = sio.loadmat("./data.mat")

class Terminal:
    def __init__(self):

        self.A = mat['A']
        self.b = mat['b']
