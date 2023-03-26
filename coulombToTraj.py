
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import os
import numpy as np
from formatData import loadData
from sklearn.model_selection import train_test_split

# Define the input and output dimensions
input_dim = 78 
output_dim = 12 * 3

def getModel():
    model = Sequential()
    model.add(Dense(units=1024, activation='linear', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(units=output_dim, activation='linear'))
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def inputFormat(raw_input : np.array):
    molRep2D_without_diagonal = []
    diagonal_index = [1,3,6,10,15,21,28,36,45,55,66,78]
    index = np.array([i-1 for i in range(1,78) if i not in diagonal_index])

    for each_molRep2D in raw_input:
        each_molRep2D_without_diagonal = each_molRep2D[index]
        molRep2D_without_diagonal.append(each_molRep2D_without_diagonal)
    molRep2D_without_diagonal = np.array(molRep2D_without_diagonal)

    # charge of all atoms in benzene
    atomic_charge = np.repeat(np.array([[6,6,6,6,6,6,1,1,1,1,1,1]]), len(raw_input), axis=0)
    molRep2D_with_atomic_charge = np.concatenate((atomic_charge, molRep2D_without_diagonal), axis=1)
    return molRep2D_with_atomic_charge

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

    molRep2D_without_diagonal = []
    diagonal_index = [1,3,6,10,15,21,28,36,45,55,66,78]
    index = np.array([i-1 for i in range(1,78) if i not in diagonal_index])

    for each_molRep2D in molRep2D:
        each_molRep2D_without_diagonal = each_molRep2D[index]
        molRep2D_without_diagonal.append(each_molRep2D_without_diagonal)
    molRep2D_without_diagonal = np.array(molRep2D_without_diagonal)

    # charge of all atoms in benzene
    atomic_charge = np.repeat(np.array([[6,6,6,6,6,6,1,1,1,1,1,1]]), len(molRep2D), axis=0)
    molRep2D_with_atomic_charge = np.concatenate((atomic_charge, molRep2D_without_diagonal), axis=1)

    flatened_xyz = np.array(getFlattenedXyz(xyz_files_dir_name))
    X_train, X_test, y_train, y_test = train_test_split(molRep2D_with_atomic_charge, flatened_xyz,
                                   test_size=0.7)
    model = getModel()
    model.fit(X_train, y_train, epochs=70)


