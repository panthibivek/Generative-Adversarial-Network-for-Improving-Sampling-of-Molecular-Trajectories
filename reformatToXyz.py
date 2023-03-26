
""" @author : Bivek
"""

import numpy as np
import os.path
import pathlib
from utils import *
from qml.representations import vector_to_matrix
from ase import Atoms
from ase.io import Trajectory

def format_xyz_samples(generated_sample : np.array, filename : str):
    """ Converts a flat xyz array to a xyz format and saves in the given file  
    """
    num_atoms = 12
    trajfile = open(filename, "w")
    trajfile.write(str(num_atoms)+"\n")
    trajfile.write("Randomly generated samples\n")
    for i in range(num_atoms):
        if i < num_atoms//2:
            atomic_symbol = "C"
        else:
            atomic_symbol = "H"
        trajfile.write(atomic_symbol + 
                    "\t{: 10.6f}".format(generated_sample[3*i]) + 
                    "\t{: 10.6f}".format(generated_sample[3*i+1]) + 
                    "\t{: 10.6f}\n".format(generated_sample[3*i+2]))
    trajfile.close()

class MapCoulomb1dToXYZ:
    def __init__(self, abs_path_xyz_dirname : str, sorted_xyz_file : str, sorted_coulomb1D_file : str):
        if not os.path.isdir(abs_path_xyz_dirname):
            raise ValueError(str(abs_path_xyz_dirname) + " Directory does not exist!")
        self.all_coulomb1D = np.loadtxt(sorted_coulomb1D_file, delimiter=',')
        self.all_traj_xyz = np.loadtxt(sorted_xyz_file, delimiter=',')
        if len(self.all_coulomb1D) != len(self.all_traj_xyz):
            raise ValueError("Size of arrays mismatch!")
        self.xyz_filename = abs_path_xyz_dirname
        self.sizeof_sample_space = len(self.all_coulomb1D)

    def generateXYZ(self, coulomb1D_arr : np.array, batch_size : int = 1000):
        for idx in range(len(coulomb1D_arr)):
            generated_xyz = self.getxyz(self.findSpecificXyzIndex(coulomb1D_arr[idx], batch_size))
            filename = self.xyz_filename + '/molecule{:06d}.xyz'.format(idx)
            format_xyz_samples(generated_xyz, filename)
    
    def getxyz(self, index : int):
        return self.all_traj_xyz[index]

    def findSpecificXyzIndex(self, each_coulomb1D_arr : np.array, batch_size : int):
        initial_search_arr = np.array([abs(sqDistFromOrigin(each_coulomb1D_arr) - sqDistFromOrigin(self.all_coulomb1D[idx])) 
                                       for idx in range(0, self.sizeof_sample_space, batch_size)])
        closest_first_search_index = np.argmin(initial_search_arr) * batch_size
        if closest_first_search_index <= batch_size:
            search_index_lower_bound = 0
            search_index_upper_bound = batch_size
        else:
            search_index_lower_bound = closest_first_search_index - batch_size
            search_index_upper_bound = closest_first_search_index + batch_size

        final_search_idx = np.array([idx2 for idx2 in range(search_index_lower_bound, search_index_upper_bound, 1)])   
        final_search_arr = np.array([sqDistSum(each_coulomb1D_arr, self.all_coulomb1D[j]) for j in final_search_idx])
        print(final_search_arr[np.argmin(final_search_arr)])
        closest_final_index = final_search_idx[np.argmin(final_search_arr)]
        return closest_final_index
        
        
def sqDistFromOrigin(arr):
    return sum(x**2 for x in arr)

def sqDistSum(arr1 : np.array, arr2 : np.array):
    diff_arr = arr1 - arr2
    return sum(x**2 for x in diff_arr)

