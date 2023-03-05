
import tensorflow as tf
from utils import random_generator, random_shuffle
import numpy as np
from formatData import loadData
from sklearn.model_selection import train_test_split

class GenAdvNetwork(tf.keras.Model):
    def __init__(self, latent_dim, batch_size) -> None:
        # initialize keras Model class
        super().__init__()
        self.count = 0
        self.latent_dim = latent_dim
        self.generator = self.generate_generator()
        self.discriminator = self.generate_discriminator()
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
    
    def generate_generator(self) -> tf.keras.Sequential:
        filters = [13, 128, 264]
        model = tf.keras.Sequential(name="generator")
        model.add(tf.keras.layers.Dense(filters[0], input_dim=self.latent_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0))
        model.add(tf.keras.layers.Reshape((filters[0], 1)))

        model.add(tf.keras.layers.Conv1DTranspose(filters[1], kernel_size=4, strides=3, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0))
        model.add(tf.keras.layers.BatchNormalization())

        for filter in filters[2:]:
            model.add(tf.keras.layers.Conv1DTranspose(filter, kernel_size=4, strides=2, padding='same'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0))
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
    
    def compile(self, generator_opt, discriminator_opt, disc_loss, gen_loss) -> None:
        super().compile(run_eagerly=True)
        self.g_optimizer = generator_opt
        self.d_optimizer = discriminator_opt
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

    def generate_trajectories(self, size_of_data):
        return self.generator(random_generator((size_of_data, self.latent_dim)))          
    
    def train_disc_gen(self, trajectories, Y_, tag):
        with tf.GradientTape() as tape:
            if tag == "discriminator":
                predictions = self.discriminator(trajectories)
                loss = self.disc_loss(Y_, predictions)
            elif tag == "generator":
                fake_trajectories = self.generator(trajectories)
                # predictions = self.discriminator(fake_trajectories)
                loss = self.gen_loss(Y_, fake_trajectories)
            else:
                raise Exception("Incorrect tag! Should be either discriminator or generator")
        if tag == "discriminator":
            gradiants = tape.gradient(loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(gradiants, self.discriminator.trainable_weights))
        else:
            gradiants = tape.gradient(loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradiants, self.generator.trainable_weights))
        return loss            

    def train_step(self, data):
        input_X, energies = data
        # energies = tf.cast(energies, dtype=tf.float64)
        size_of_data_ = int(len(input_X)) # note that this value may change in the last batch
        input_X = tf.cast(input_X, dtype=tf.float64)

        # generating trajectories
        generated_trajectories = self.generate_trajectories(size_of_data_)
        generated_trajectories = tf.cast(generated_trajectories, dtype=tf.float64)
        combined_trajectories = tf.concat([generated_trajectories, input_X], axis=0)

        # labels for differentiating real vs fake trajectories
        combined_label = tf.concat([tf.zeros((size_of_data_, 1)), tf.ones((size_of_data_, 1))], axis=0)

        # Shuffling the inputs randomly
        combined_trajectories, combined_label = random_shuffle(combined_trajectories, combined_label)

        # Training the discriminator.
        d_loss = self.train_disc_gen(trajectories=combined_trajectories, Y_=combined_label, tag="discriminator")

        # Generating random labels and concinate with real energies
        random_vector_labels = random_generator((size_of_data_, self.latent_dim))
        # misleading_labels = tf.zeros((size_of_data_, 1))
        # Training the generator
        g_loss = self.train_disc_gen(trajectories=random_vector_labels, Y_=input_X, tag="generator")
        # Monitoring loss
        self.discriminator_loss.update_state(d_loss)
        self.generator_loss.update_state(g_loss)
        return {
            "g_loss": self.generator_loss.result(),
            "d_loss": self.discriminator_loss.result(),
        }

if __name__=="__main__":
    pass