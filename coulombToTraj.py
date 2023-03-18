
import keras
from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np

# Define the input and output dimensions
input_dim = 78
output_dim = 12 * 3

def getModel():
    # Define the MLP architecture
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_dim, activation='linear'))

    # Compile the model with mean squared error loss and Adam optimizer
    model.compile(loss='mse', optimizer='adam')

    # Print a summary of the model architecture
    model.summary()

def extractDataFromXyz(filename : str):
    with open(filename) as f:
        lines = f.readlines()
    data_vector = []
    for line in lines[2:]:
        fields = line.split()
        data_vector.extend([float(field) for field in fields[1:]])
    return data_vector

def getFlattenedXyz(dir_name : str):
    list_flattened_xyz = []
    list_of_xyz_files =  sorted([dir_name + xyzfile for xyzfile in os.listdir(dir_name)])
    for xyz_file in list_of_xyz_files:
        list_flattened_xyz.append(extractDataFromXyz(xyz_file))
    return list_flattened_xyz

if __name__=="__main__":
    dir_name = "/home/panthibivek/thesis/GAN_pkg/data/AllMolecules/"
    flattened_xyz = np.array(getFlattenedXyz(dir_name))
    np.savetxt("/home/panthibivek/thesis/GAN_pkg/data/flattenedXyz.txt", flattened_xyz, delimiter=',')
