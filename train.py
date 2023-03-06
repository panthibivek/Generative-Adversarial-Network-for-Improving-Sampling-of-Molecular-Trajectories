
""" @author : Bivek Panthi
"""
import tensorflow as tf
import numpy as np
from formatData import loadData
from sklearn.model_selection import train_test_split
from gan import GenAdvNetwork
import os
from matplotlib import pyplot as plt

latent_dim_ = 78
epochs_ = 1
batch_size_ = 32
trajectory_size = 78

if __name__=="__main__":
    """
    Loading data
    Note than we can only specify absolute location of the raw data
    """
    molRep2D, energies = loadData(12, "/home/panthibivek/thesis/GAN_pkg/data/traj.xyz")
    #split it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(molRep2D,energies, test_size=0.1)

    print("Training data size:", X_train.shape)
    print("Test data size:", X_test.shape)

    y_train = np.reshape(y_train, (-1, 1))
    X_train = np.array(X_train)
    X_train = X_train.astype(float)
    X_train = np.reshape(X_train, (-1, trajectory_size, 1))
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=100).batch(batch_size_)

    GAN_model = GenAdvNetwork(latent_dim=latent_dim_, batch_size=batch_size_)
    GAN_model.compile(
        generator_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
        discriminator_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
        disc_loss=tf.keras.losses.BinaryCrossentropy(),
        gen_loss=tf.keras.losses.MAE
    )
    history = GAN_model.fit(dataset, epochs=epochs_)

    train_dir = os.path.dirname(os.path.abspath("__file__")) + "/runs/train/"
    only_dir = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    if len(only_dir) > 0:
        last_train_seq_number =  int(only_dir[-1][-1])
    else:
        last_train_seq_number = 0
    current_train_dir = train_dir + "exp" + str(last_train_seq_number+1)
    os.mkdir(current_train_dir)
    GAN_model.save_weights(current_train_dir + "/weights.h5")

    plt.plot(history.history['d_loss'])
    plt.title('Discriminator Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(current_train_dir + '/disLoss.png', dpi = 300)

    plt.plot(history.history['g_loss'])
    plt.title('Generator Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(current_train_dir + '/genLoss.png', dpi = 300)