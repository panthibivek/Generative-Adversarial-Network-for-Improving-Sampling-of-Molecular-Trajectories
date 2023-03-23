
""" @author : Bivek
"""

import pathlib
import qml
import numpy as np
import os
import json

class Generatexyz:
    def __init__(self, total_samples_to_generate : int, batch_size : int, number_of_atoms : int):
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
        mol = qml.Compound(xyz=self.temp_xyz)
        mol.generate_coulomb_matrix(size=self.number_of_atoms, sorting="row-norm")
        return mol.representation      

    def format_xyz_samples(self, generated_sample : np.array):
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
        return np.random.normal(0, 0.005, size_n)

    def sorting_by_coulomb_matrix(self):
        all_coulomb1D = np.loadtxt(self.coulomb1D_dirname, delimiter=',')
        all_traj_xyz = np.loadtxt(self.traj_dirname, delimiter=',')
        idx = list(range(len(all_coulomb1D)))
        idx.sort(key=lambda i: sqDistFromOrigin(all_coulomb1D[i]))
        sorted_all_coulomb1D = np.array(list(map(all_coulomb1D.__getitem__, idx)))
        sorted_all_traj_xyz = np.array(list(map(all_traj_xyz.__getitem__, idx)))

        sorted_traj_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalSortedTraj.txt"
        sorted_traj_file = open(sorted_traj_dirname, "w")
        np.savetxt(sorted_traj_file, sorted_all_traj_xyz, delimiter=',')

        sorted_coulomb1D_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/exp/totalSortedCoulomb1D.txt"
        sorted_coulomb1D_file = open(sorted_coulomb1D_dirname, "w")
        np.savetxt(sorted_coulomb1D_file, sorted_all_coulomb1D, delimiter=',')

def sqDistFromOrigin(arr):
    return sum(x**2 for x in arr)