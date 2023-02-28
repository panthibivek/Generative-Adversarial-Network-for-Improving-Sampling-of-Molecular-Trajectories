
""" @author : Bivek
"""

import qml
import numpy as np
import os.path
import pathlib
from utils import *

class GenerateCsv:
    def __init__(self, max_size : int, filename : str):
        try:
            int(max_size)
        except ValueError:
            raise ValueError("Size must be integer!")
        if os.path.splitext(filename)[1] != ".xyz":
            raise ValueError("file type must be xyz format!")
        self.max_size = max_size
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        self.XYZdirname = self.current_dir + '/data/AllMolecules'
        if os.path.isdir(self.XYZdirname):
            pass
        else:
            self.GenerateXYZFiles(filename)

        if os.path.isfile(self.current_dir + "/data/lower_coulomb_mtx_array.txt"):
            pass
        else:
            self.CreateCoulombMtx()

    def GenerateXYZFiles(self, file_name : str):
        total_molecules = buf_count_newlines(file_name) // 14
        new_dir = pathlib.Path(self.current_dir + '/data/', "AllMolecules")
        new_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.current_dir + '/data/AllMolecules')
        for i in range(total_molecules):
            ExtractLines(file_name, i)
        os.chdir(self.current_dir) #back to original location

    def CreateCoulombMtx(self):
        lower_coulomb_mtx_array = []
        os.chdir(self.current_dir + '/data/AllMolecules')
        for path in sorted(os.listdir(self.XYZdirname)):
            path_in_str = os.path.abspath(str(path))
            mol = qml.Compound(xyz=path_in_str)
            mol.generate_coulomb_matrix(size=self.max_size, sorting="row-norm")
            lower_coulomb_mtx_array.append(list(mol.representation))
        os.chdir(self.current_dir)
        lower_coulomb_mtx_array = np.array(lower_coulomb_mtx_array)
        mtx_filename = self.current_dir + "/data/lower_coulomb_mtx_array.txt"
        np.savetxt(mtx_filename, lower_coulomb_mtx_array, delimiter=",")
        return lower_coulomb_mtx_array

def loadData(max_size : int, filename : str):
    #Only abs path accepted
    obj = GenerateCsv(max_size, filename)
    ###################################
    molRep2D = []
    current_dir = os.path.abspath(os.path.dirname(__file__))
    with open(current_dir + "/data/lower_coulomb_mtx_array.txt", 'r') as f:
        for line in f.readlines():
            molRep2D.append(line.split(','))
    molRep2D = np.array(molRep2D)
    print("input data size:", molRep2D.shape)
    ###################################
    energies = []
    with open(current_dir + "/data/energies.txt", 'r') as f:
        energies = [float(line.strip()) for line in f if line]
    energies = np.array(energies)
    print("output data size:", energies.shape)
    ###################################
    return molRep2D, energies

if __name__=="__main__":
    #Only abs path accepted
    molRep2D, energies = loadData(12, "/home/panthibivek/thesis/GAN_pkg/data/traj.xyz")
    pass