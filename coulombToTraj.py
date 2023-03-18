
import keras
from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np
from formatData import loadData
from sklearn.model_selection import train_test_split

# Define the input and output dimensions
input_dim = 78
output_dim = 12 * 3

def getModel():
    # Define the MLP architecture
    model = Sequential()
    model.add(Dense(units=1024, activation='linear', input_dim=input_dim))
    model.add(Dense(units=512, activation='linear'))
    model.add(Dense(units=256, activation='linear'))
    model.add(Dense(units=64, activation='linear'))
    model.add(Dense(units=32, activation='linear'))
    model.add(Dense(units=output_dim, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def extractDataFromXyz(filename : str):
    with open(filename) as f:
        lines = f.readlines()
    data_vector = []
    for line in lines[2:]:
        fields = line.split()
        data_vector.extend([float(field) for field in fields[1:]])
    return data_vector

def getFlattenedXyz(dir_name : str):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    flattened_xyz_filename = current_dir + '/data/flattenedXyz.txt'
    if os.path.isfile(flattened_xyz_filename):
        return np.loadtxt(flattened_xyz_filename, delimiter=',', dtype=float)
    else:
        list_flattened_xyz = []
        list_of_xyz_files =  sorted([os.path.abspath(dir_name + xyzfile) for xyzfile in os.listdir(dir_name)])
        for xyz_file in list_of_xyz_files:
            list_flattened_xyz.append(extractDataFromXyz(xyz_file))
        np.savetxt(flattened_xyz_filename, list_flattened_xyz, delimiter=',')
        return np.array(list_flattened_xyz)

if __name__=="__main__":
    xyz_files_dir_name = "/home/panthibivek/thesis/GAN_pkg/data/AllMolecules/"
    xyz_traj_filename = "/home/panthibivek/thesis/GAN_pkg/data/traj.xyz"
    print("Loading Lower Coulomb representation and respective energies")
    molRep2D, _ = loadData(12, xyz_traj_filename)
    flatened_xyz = np.array(getFlattenedXyz(xyz_files_dir_name))
    X_train, X_test, y_train, y_test = train_test_split(molRep2D, flatened_xyz,
                                   test_size=0.5)
    model = getModel()
    model.fit(X_train, y_train, epochs=100)

"""
To the @author
Problem           : Model is not learning even if increase the number of parameters.

Possible Solution : Change the model architecture
"""


