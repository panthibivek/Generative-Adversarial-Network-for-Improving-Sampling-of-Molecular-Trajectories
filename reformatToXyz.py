
""" @author : Bivek
"""

import numpy as np
import os.path
import pathlib
from utils import *
from qml.representations import vector_to_matrix

class ReformatToXYZ:
    def __init__(self, lower_tri_coulomb_mtx : np.array):
        # Assuming that all inputs in lower_tri_coulomb_mtx are for Benzene
        if not isinstance(lower_tri_coulomb_mtx, np.ndarray):
          raise TypeError("Passed array must be numpy!")
        self.lower_tri_coulomb_mtx = lower_tri_coulomb_mtx

    def manager(self):
        for each_coulomb_mtx in self.lower_tri_coulomb_mtx:
            return self.retransform_1d_to_2d(each_coulomb_mtx)
    
    def retransform_1d_to_2d(self, coulomb_mtx_1d):
        reshaped_arr = np.reshape(coulomb_mtx_1d, (78,))
        return vector_to_matrix(reshaped_arr)