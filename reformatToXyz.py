
""" @author : Bivek
"""

import numpy as np
import os.path
import pathlib
from utils import *
from qml.representations import vector_to_matrix
from scipy.spatial import KDTree


def format_xyz_samples(generated_sample : np.array, filename : str):
    """ Converts a flat xyz array to a xyz format and saves in the given file  

    Parameters : 
        filename            : name of the file to store the xyz file
        generated_sample    : content of the file
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
    """  This class is used to construct a KD tree and map each Coulomb matrix to 3D molecular coordinates
    """
    def __init__(self, abs_path_xyz_dirname : str, sorted_xyz_file : str, sorted_coulomb1D_file : str):
        """  Class constructor. This method also constructs the KD tree.

        Parameters : 
        abs_path_xyz_dirname    : absolute path of xyz file 
        sorted_xyz_file         : file to store the sorted xyz file
        sorted_coulomb1D_file   : file to store the coulomb matrix vector
        """
        if not os.path.isdir(abs_path_xyz_dirname):
            raise ValueError(str(abs_path_xyz_dirname) + " Directory does not exist!")
        self.all_coulomb1D = np.loadtxt(sorted_coulomb1D_file, delimiter=',')
        self.all_traj_xyz = np.loadtxt(sorted_xyz_file, delimiter=',')
        if len(self.all_coulomb1D) != len(self.all_traj_xyz):
            raise ValueError("Size of arrays mismatch!")
        self.xyz_filename = abs_path_xyz_dirname
        self.sizeof_sample_space = len(self.all_coulomb1D)
        self.tree = KDTree(self.all_coulomb1D)

    def generateXYZ(self, coulomb1D_arr : np.array):
        """ Function that takes the array of Coulomb matrces, finds the 3D molecular coordinates 
            for each molecule and reformats them to XYZ file format

        Parameters : 
        coulomb1D_arr           : numpy arrays to find the 3D coordinates 
        """
        list_return = []
        for idx in range(len(coulomb1D_arr)):
            # generated_xyz = self.getxyz(self.findSpecificXyzIndex(coulomb1D_arr[idx]))
            # list_return += list(self.findSpecificXyzIndex(coulomb1D_arr[idx]))
            list_return.append(self.findSpecificXyzIndex(coulomb1D_arr[idx]))
        list_abc = []
        for i in range(len(list_return)):
            filename = self.xyz_filename + '/molecule{:06d}.xyz'.format(i)
            # format_xyz_samples(self.all_traj_xyz[list_return[i]], filename)
            list_abc.append(self.all_traj_xyz[list_return[i]])
        return list_abc
    
    def getxyz(self, index : int):
        """  Function that returns the 3D molecular coordinates given an index

        Parameters : 
        index               : index of the molecular coordinate in the sample space
        """
        return self.all_traj_xyz[index]

    def findSpecificXyzIndex(self, each_coulomb1D_arr : np.array):
        """  Function that querys the index of the nearest 3D molecular coordinates for a Coulomb matrix

        Parameters : 
        each_coulomb1D_arr  : specific array to find the 3D molecular coordinate
        """
        dist, index = self.tree.query(each_coulomb1D_arr, k=1)
        return index
        
        
def sqDistFromOrigin(arr):
    """ Function to find the squared distance of an array from origin 

    Parameters : 
        arr                 : array for which the distance should be calculated
    """
    return sum(x**2 for x in arr)

def sqDistSum(arr1 : np.array, arr2 : np.array):
    """ Function to find the sum of square the difference of arrays

    Parameters : 
        arr1                 : first input array 
        arr2                 : second input array 
    """
    diff_arr = arr1 - arr2
    return sum(x**2 for x in diff_arr)

