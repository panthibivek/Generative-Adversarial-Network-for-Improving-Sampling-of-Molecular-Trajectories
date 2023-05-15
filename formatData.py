
""" @author : Bivek
"""

import qml
import numpy as np
import os.path
import pathlib
from utils import *

class GenerateCsv:
    def __init__(self, max_size : int, filename : str, xyzdirname : str, coulombMtxFilename : str):
        """ Class constructor

        Parameters : 
        max_size                : total number of atoms in the molecules
        filename                : absolute path to the XYZ file
        xyzdirname              : directory where all the individual XYZ files for each molecules are stored
                                  Only path from GAN package
        coulombMtxFilename      : filename to save generated 1D representation of coulomb matrix
                                  Only path from GAN package
        """
        try:
            int(max_size)
        except ValueError:
            raise ValueError("Size must be integer!")
        if os.path.splitext(filename)[1] != ".xyz":
            raise ValueError("file type must be xyz format!")
        self.max_size = max_size
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        self.XYZdirname = self.current_dir + xyzdirname
        self.mtx_filename = self.current_dir + coulombMtxFilename
        if os.path.isdir(self.XYZdirname):
            pass
        else:
            self.GenerateXYZFiles(filename)

        if os.path.isfile(self.mtx_filename):
            pass
        else:
            self.CreateCoulombMtx()

    def GenerateXYZFiles(self, file_name : str): 
        """ Function that seperates the molecules into individual XYZ files

        Parameters : 
        filename                : absolute path to the XYZ file
        """
        total_molecules = buf_count_newlines(file_name) // 14
        new_dir = pathlib.Path(self.XYZdirname)
        new_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.XYZdirname)
        for i in range(total_molecules):
            ExtractLines(file_name, i)
        os.chdir(self.current_dir) #back to original location

    def CreateCoulombMtx(self):
        """ Function that generates Coulomb matrix for each molecules
        """
        lower_coulomb_mtx_array = []
        os.chdir(self.XYZdirname)
        for path in sorted(os.listdir(self.XYZdirname)):
            path_in_str = os.path.abspath(str(path))
            mol = qml.Compound(xyz=path_in_str)
            mol.generate_coulomb_matrix(size=self.max_size, sorting="unsorted")
            lower_coulomb_mtx_array.append(list(mol.representation))
        os.chdir(self.current_dir)
        lower_coulomb_mtx_array = np.array(lower_coulomb_mtx_array)
        np.savetxt(self.mtx_filename, lower_coulomb_mtx_array, delimiter=",")
        return lower_coulomb_mtx_array

def loadData(max_size : int, filename : str, 
             energyFilename : str = os.path.abspath(os.path.dirname(__file__)) + "/data/energies.txt", 
             xyzdirname : str = "/data/AllMolecules", 
             coulombMtxFilename : str = "/data/lower_coulomb_mtx_array.txt"):
    """
    Parameters : 
        max_size            : max atom numbers
        filename            : filename of trajectory
                                    Only abs path accepted
        energyFilename      : filename of energy
                                    Only abs path accepted
        xyzdirname          : dirname where xyz will be saved
                                    Only path from GAN package
        coulombMtxFilename  : filename to save generated 1D representation of coulomb matrix
                                    Only path from GAN package

    """
    #Only abs path accepted
    obj = GenerateCsv(max_size, filename, xyzdirname, coulombMtxFilename)
    ###################################
    current_dir = os.path.abspath(os.path.dirname(__file__))
    molRep2D = np.loadtxt(current_dir + coulombMtxFilename, delimiter=',', dtype=float)
    print("input data size:", molRep2D.shape)
    ###################################
    energies = []
    with open(energyFilename, 'r') as f:
        energies = [float(line.strip()) for line in f if line]
    energies = np.array(energies)
    print("output data size:", energies.shape)
    ###################################
    return molRep2D, energies

if __name__=="__main__":
    # molRep2D, energies = loadData(max_size=12, 
    #                               filename="/home/panthibivek/thesis/GAN_pkg/data/traj.xyz", 
    #                               energyFilename="/home/panthibivek/thesis/GAN_pkg/data/energies.txt",
    #                               xyzdirname="/data/test_exp/AllMolecules",
    #                               coulombMtxFilename="/data/test_exp/lower_coulomb_mtx_array.txt")
    molRep2D, energies = loadData(max_size=12, 
                                filename="/home/panthibivek/thesis/GAN_pkg/data/traj.xyz")
    print(molRep2D)