import scipy.io as sio

zf_set = sio.loadmat("./src/set/Zf_set_new.mat")
z_set = sio.loadmat("./src/set/Z_set_new.mat")

def ensure_nx3(A):
    if A.shape[1] == 3:
        return A
    elif A.shape[0] == 3:
        return A.T
    elif A.size % 3 == 0:
        return A.reshape(-1, 3)
    else:
        raise ValueError(f"Cannot reshape A to nx3 form. Got shape {A.shape}")

class ZfSet:
    def __init__(self):
        self.A = ensure_nx3(zf_set['A'])
        self.b = zf_set['b']

class ZSet:
    def __init__(self):
        self.A = ensure_nx3(z_set['A'])
        self.b = z_set['b']
