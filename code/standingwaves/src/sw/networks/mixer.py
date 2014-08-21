import numpy as np

def Mixer(rx, ry, tx, ty, rot):
    Rd = rot
    Ri = Rd.T
    S11 = np.array([[rx, 0, 0],
                    [0, ry, 0],
                    [0, 0, 0]], dtype=complex)
    S12 = np.zeros((3, 3))
    S21 = np.array([[tx, 0, 0],
                    [0, ty, 0],
                    [0, 0, 0]], dtype=complex)
    S22 = np.zeros((3, 3))
    S11 = Rd.dot(S11).dot(Ri)
    S21 = Rd.dot(S21).dot(Ri)
    return np.array(np.bmat([[S11, S12],
                             [S21, S22]]))
