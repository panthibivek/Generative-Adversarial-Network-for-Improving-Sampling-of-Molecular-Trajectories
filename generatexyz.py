
""" @author : Bivek
"""

import pathlib
import qml
import numpy as np
import os
import json

class Generatexyz:
    """ This class is used to generate large sample of molecules for KD Tree search algorithm
    """
    def __init__(self, total_samples_to_generate : int, batch_size : int, number_of_atoms : int):
        """ Class constructor

        Parameters : 
        total_samples_to_generate       : total number of samples to generate for sample space 
        batch_size                      : size of batch after which the samples are stored in a file
        number_of_atoms                 : total number of atoms in the molecule. for example for Benzene the value must be 12
        """
        self.total_samples_to_generate = total_samples_to_generate
        self.batch_size = batch_size
        self.number_of_atoms = number_of_atoms

        new_dir = pathlib.Path(os.path.abspath(os.path.dirname(__file__)) + "/data/exp")
        new_dir.mkdir(parents=True, exist_ok=True)
        self.traj_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalTraj.txt"
        self.coulomb1D_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalCoulomb1D.txt"
        self.cache_memory = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/cacheData.json"
        self.temp_xyz = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/tempXyz.xyz"
        if os.path.isfile(self.cache_memory):
            pass
        else:
            datadict = {
                "totalBatch" : self.total_samples_to_generate//self.batch_size,
                "currentBatch" : 0
            }
            json_obj = json.dumps(datadict, indent=2)
            with open(self.cache_memory, "w") as outfile:
                outfile.write(json_obj)
    
    def generate_samples(self, benzene_traj : np.array):
        """ Function that generates specified number of 3D molecular coordinates

        Parameters : 
        benzene_traj       : 3D coordinates of a real molecule used to create the sample space
        """
        with open(self.cache_memory, 'r') as openfile:
            json_obj = json.load(openfile)
        if int(json_obj["currentBatch"]) < int(json_obj["totalBatch"]):
            for idx in range(int(json_obj["currentBatch"]), int(json_obj["totalBatch"])):
                print("Current Batch: ", idx)
                self.generate_samples_batch(benzene_traj)
                json_obj["currentBatch"] = idx + 1
                with open(self.cache_memory, 'w') as openfile:
                    json.dump(json_obj, openfile)         

    def generate_samples_batch(self, benzene_traj : np.array):
        """ Function that generates one batch size of 3D molecular coordinates

        Parameters : 
        benzene_traj       : 3D coordinates of a real molecule used to create the sample space
        """
        size_ = len(benzene_traj)
        temp_generated_samples = []
        temp_generated_coulomb1D = []
        for i in range(self.batch_size):
            gaussian_noise = self.generate_gaussian_noise(size_)
            new_traj = gaussian_noise + benzene_traj
            temp_generated_samples.append(new_traj)
            self.format_xyz_samples(new_traj)
            temp_generated_coulomb1D.append(list(self.generate_coulomb_matrix()))
        temp_generated_samples = np.array(temp_generated_samples)
        trajfile = open(self.traj_dirname, "a")
        np.savetxt(trajfile, temp_generated_samples, delimiter=',')
        trajfile.close()

        temp_generated_coulomb1D = np.array(temp_generated_coulomb1D)
        coulomb1Dfile = open(self.coulomb1D_dirname, "a")
        np.savetxt(coulomb1Dfile, temp_generated_coulomb1D, delimiter=',')
        coulomb1Dfile.close()

    def generate_coulomb_matrix(self):
        """  Function that generates the Coulomb matrx for all molecules
        """
        mol = qml.Compound(xyz=self.temp_xyz)
        mol.generate_coulomb_matrix(size=self.number_of_atoms, sorting="unsorted")
        return mol.representation      

    def format_xyz_samples(self, generated_sample : np.array):
        """ Function that reformats the newly generated molecules to the XYZ file format

        Parameters : 
        generated_sample     : flattened 3D coordinates of the molecule
        """
        trajfile = open(self.temp_xyz, "w")
        trajfile.write(str(self.number_of_atoms)+"\n")
        trajfile.write("Randomly generated samples\n")
        for i in range(self.number_of_atoms):
            if i < self.number_of_atoms//2:
                atomic_symbol = "C"
            else:
                atomic_symbol = "H"
            trajfile.write(atomic_symbol + 
                        "\t{: 10.6f}".format(generated_sample[3*i]) + 
                        "\t{: 10.6f}".format(generated_sample[3*i+1]) + 
                        "\t{: 10.6f}\n".format(generated_sample[3*i+2]))
        trajfile.close()

    def generate_gaussian_noise(self, size_n):
        """ Function that generates Gaussian noise

        Parameters : 
        size_n              : size of gaussian noise to generate
        """
        return np.random.normal(0, 0.005, size_n)

    def sorting_by_coulomb_matrix(self):
        """ Function that sorts all the molecules by the squared distance of each 
            Coulomb matrix vector from origin in ascending order
        """
        all_coulomb1D = np.loadtxt(self.coulomb1D_dirname, delimiter=',')
        all_traj_xyz = np.loadtxt(self.traj_dirname, delimiter=',')
        idx = list(range(len(all_coulomb1D)))
        idx.sort(key=lambda i: sqDistFromOrigin(all_coulomb1D[i]))
        sorted_all_coulomb1D = np.array(list(map(all_coulomb1D.__getitem__, idx)))
        sorted_all_traj_xyz = np.array(list(map(all_traj_xyz.__getitem__, idx)))

        self.sorted_traj_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalSortedTraj.txt"
        with open(self.sorted_traj_dirname, "w") as f:
            for row in sorted_all_traj_xyz:
                f.write(','.join(str(x) for x in row) + '\n')

        self.sorted_coulomb1D_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalSortedCoulomb1D.txt"
        with open(self.sorted_coulomb1D_dirname, "w") as f:
            for row in sorted_all_coulomb1D:
                f.write(','.join(str(x) for x in row) + '\n')

def sqDistFromOrigin(arr):
    """ Function to find the sq. distance of an array from origin

    Parameters : 
    arr                      : numpy array whose distance to calculate
    """
    return sum(x**2 for x in arr)