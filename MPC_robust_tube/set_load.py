import numpy as np
import scipy.io as sio

Zf_mat = sio.loadmat("./Zf_set.mat")
Z_mat = sio.loadmat("./Z_set.mat")

def ensure_nx3(A):
    if A.shape[1] == 3:
        return A
    elif A.shape[0] == 3:
        return A.T
    elif A.size % 3 == 0:
        return A.reshape(-1, 3)
    else:
        raise ValueError(f"Cannot reshape A to nx3 form. Got shape {A.shape}")

class Zf_set:
    def __init__(self):
        self.A = ensure_nx3(Zf_mat['A'])
        self.b = Zf_mat['b']

class Z_set:
    def __init__(self):
        self.A = ensure_nx3(Z_mat['A'])
        self.b = Z_mat['b']
