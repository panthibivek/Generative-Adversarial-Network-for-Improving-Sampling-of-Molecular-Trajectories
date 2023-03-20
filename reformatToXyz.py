
""" @author : Bivek
"""

import numpy as np
import os.path
import pathlib
from utils import *
from qml.representations import vector_to_matrix
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write
from ase.io import Trajectory

class ReformatToXYZ:
    def __init__(self, lower_tri_coulomb_mtx : np.array):
        # Assuming that all inputs in lower_tri_coulomb_mtx are for Benzene
        if not isinstance(lower_tri_coulomb_mtx, np.ndarray):
          raise TypeError("Passed array must be numpy!")
        self.atomic_nums = [6,6,6,6,6,6,1,1,1,1,1,1]
        self.lower_tri_coulomb_mtx = lower_tri_coulomb_mtx

    def calculateXyz(self):
        for each_coulomb_mtx in self.lower_tri_coulomb_mtx:
            coulomb2D = self.retransform_1d_to_2d(each_coulomb_mtx)
            xyz = self.coulomb_matrix_to_xyz(coulomb2D)
        return xyz
    
    def retransform_1d_to_2d(self, coulomb_mtx_1d):
        reshaped_arr = np.reshape(coulomb_mtx_1d, (78,))
        return np.array(vector_to_matrix(reshaped_arr))

    def coulomb_matrix_to_xyz(self, coulomb_matrix):
        # Create Atoms object
        atoms = Atoms(numbers=self.atomic_nums, positions=None)

        # Set atomic positions
        from scipy.spatial.distance import squareform
        from scipy.spatial.distance import pdist

        distance_matrix = squareform(pdist(coulomb_matrix))
        print(distance_matrix)
        atoms.set_positions(distance_matrix)

        # Create trajectory and add frame
        traj = Trajectory('output.xyz', 'w', atoms)
        traj.write(atoms)

        return atoms

