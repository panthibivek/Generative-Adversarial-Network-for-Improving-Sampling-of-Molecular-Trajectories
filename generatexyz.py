
import numpy as np
import os
import json

def sqDistFromOrigin(arr):
    return sum(x**2 for x in arr)

# myListOfVectors.sort(key=sqDistFromOrigin)

class Generatexyz:
    def __init__(self, total_samples_to_generate : int, batch_size : int):
        self.total_samples_to_generate = total_samples_to_generate
        self.batch_size = batch_size
        self.traj_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/totalSortedTraj.txt"
        self.coulomb1D_dirname = os.path.abspath(os.path.dirname(__file__)) + "/data/totalCoulomb1D.txt"
        self.cache_memory = os.path.abspath(os.path.dirname(__file__)) + "/data/cacheData.json"
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
        for i in range(self.batch_size):
            gaussian_noise = self.generate_gaussian_noise(size_)
            temp_generated_samples.append(gaussian_noise + benzene_traj)
        temp_generated_samples = np.array(temp_generated_samples)
        trajfile = open(self.traj_dirname, "a")
        np.savetxt(trajfile, temp_generated_samples)
        trajfile.close()

    def generate_gaussian_noise(self, size_n):
        return np.random.normal(0, 0.005, size_n)