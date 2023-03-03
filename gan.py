
import tensorflow as tf
from utils import random_generator
import numpy as np

from formatData import loadData
from sklearn.model_selection import train_test_split

class GenAdvNetwork(tf.keras.Model):
    def __init__(self, latent_dim, batch_size) -> None:
        # initialize keras Model class
        super().__init__()
        self.count = 0
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.generator = self.generate_generator()
        self.discriminator = self.generate_discriminator()
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
    
    def generate_generator(self) -> tf.keras.Sequential:
        filters = [13, 512, 1024]
        model = tf.keras.Sequential(name="generator")
        model.add(tf.keras.layers.Dense(filters[0], input_dim=self.latent_dim+1))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Reshape((filters[0], 1)))

        model.add(tf.keras.layers.Conv1DTranspose(filters[1], kernel_size=4, strides=3, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization())

        for filter in filters[2:]:
            model.add(tf.keras.layers.Conv1DTranspose(filter, kernel_size=4, strides=2, padding='same'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(1, kernel_size=4, padding='same', activation='tanh'))
        model.summary()
        return model

    def generate_discriminator(self) -> tf.keras.Model:
        filters = [64, 128]
        model = tf.keras.Sequential(name="discriminator")
        model.add(tf.keras.layers.Conv1D(filters[0], kernel_size=4, strides=3, input_shape=(78, 1), padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        for filter in filters[1:]:
            model.add(tf.keras.layers.Conv1D(filter, kernel_size=4, strides=2, padding='same'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary()
        return model
    
    def compile(self, generator_opt, discriminator_opt, loss_func) -> None:
        super().compile(run_eagerly=True)
        self.g_optimizer = generator_opt
        self.d_optimizer = discriminator_opt
        self.loss_fn = loss_func

    def generate_trajectories(self, energies):
        return self.generator(random_generator((int(len(energies)), self.latent_dim), energies))
    
    def train_disc_gen(self, trajectories, energy_labels, tag):
        with tf.GradientTape() as tape:
            if tag == "discriminator":
                predictions = self.discriminator(trajectories)
            elif tag == "generator":
                fake_trajectories = self.generator(trajectories)
                predictions = self.discriminator(fake_trajectories)
            else:
                raise Exception("Incorrect tag! Should be either discriminator or generator")
            loss = self.loss_fn(energy_labels, predictions)
        if tag == "discriminator":
            gradiants = tape.gradient(loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(gradiants, self.discriminator.trainable_weights))
        else:
            gradiants = tape.gradient(loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradiants, self.generator.trainable_weights))
        return loss            

    def train_step(self, data):
        input_X, energies = data
        energies = tf.cast(energies, dtype=tf.float64)
        size_of_data_ = int(len(energies)) # note that this value may change in the last batch
        input_X = tf.cast(input_X, dtype=tf.float64)

        # generating trajectories
        generated_trajectories = self.generate_trajectories(energies)
        generated_trajectories = tf.cast(generated_trajectories, dtype=tf.float64)
        combined_trajectories = tf.concat([generated_trajectories, input_X], axis=0)



        # temp = tf.reshape(generated_trajectories, (-1, 78)) 
        # self.count += 1
        # file_name = "/home/panthibivek/thesis/GAN_pkg/demo_storage/" + str(self.count) + "_file.txt"
        # np.savetxt(file_name, temp.numpy())



        # labels for differentiating real vs fake trajectories
        combined_energies_label = tf.concat([tf.zeros((size_of_data_, 1)), tf.ones((size_of_data_, 1))], axis=0)
        # Training the discriminator.
        d_loss = self.train_disc_gen(trajectories=combined_trajectories, energy_labels=combined_energies_label, tag="discriminator")

        # Generating random labels and concinate with real energies
        random_vector_labels = random_generator((size_of_data_, self.latent_dim), energies)
        misleading_labels = tf.ones((size_of_data_, 1))
        # Training the generator
        g_loss = self.train_disc_gen(trajectories=random_vector_labels, energy_labels=misleading_labels, tag="generator")
        # Monitoring loss
        self.discriminator_loss.update_state(d_loss)
        self.generator_loss.update_state(g_loss)
        return {
            "g_loss": self.generator_loss.result(),
            "d_loss": self.discriminator_loss.result(),
        }

if __name__=="__main__":
    """
    Loading data
    Note than we can only specify absolute location of the raw data
    """
    molRep2D, energies = loadData(12, "/home/panthibivek/thesis/GAN_pkg/data/traj.xyz")
    #randomize the data and split it into training and test set
    # unison_shuffle(molRep2D, energies)
    X_train, X_test, y_train, y_test = train_test_split(molRep2D,energies,
                                    random_state=104, 
                                    test_size=0.3, 
                                    shuffle=True)

    print("Training data size:", X_train.shape)
    print("Test data size:", X_test.shape)

    latent_dim_ = 78
    epochs_ = 10
    batch_size_ = 64
    y_train = np.reshape(y_train, (-1, 1))
    X_train = np.array(X_train)
    X_train = X_train.astype(float)
    X_train = np.reshape(X_train, (-1, 78, 1))
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=100).batch(batch_size_)

    GAN_model = GenAdvNetwork(latent_dim=latent_dim_, batch_size=batch_size_)
    GAN_model.compile(
        generator_opt=tf.keras.optimizers.Adam(learning_rate=0.01),
        discriminator_opt=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss_func=tf.keras.losses.mean_absolute_error,
    )
    GAN_model.fit(dataset, epochs=epochs_)