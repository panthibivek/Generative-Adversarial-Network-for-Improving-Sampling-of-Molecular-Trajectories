
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
        self.manager()

    def manager(self):
        for each_coulomb_mtx in self.lower_tri_coulomb_mtx:
            print(self.retransform_1d_to_2d(each_coulomb_mtx))
    
    def retransform_1d_to_2d(self, coulomb_mtx_1d):
        reshaped_arr = np.reshape(coulomb_mtx_1d, (78,))
        return vector_to_matrix(reshaped_arr)

# from qmmlpack import AtomicCoordinatesConverter

# # Generate a random 3D Cartesian coordinate configuration for a water molecule
# coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# # Calculate the Coulomb matrix for the water molecule
# coulomb = generate_coulomb_matrix(coords, nuclear_charges=[8, 1, 1])

# # Create an AtomicCoordinatesConverter object and use it to convert the Coulomb matrix to Cartesian coordinates
# converter = AtomicCoordinatesConverter(method='coulomb_matrix', representation='cartesian')
# new_coords = converter.inverse_transform(coulomb)

# # Print the new Cartesian coordinates
# print(new_coords)