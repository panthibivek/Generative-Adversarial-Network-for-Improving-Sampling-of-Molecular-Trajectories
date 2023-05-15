
""" @author : Bivek Panthi
"""

import tensorflow as tf
from utils import random_generator, random_shuffle
import os

class GenAdvNetwork(tf.keras.Model):
    """ This class is used to compile and train the GAN model.
    """
    def __init__(self, latent_dim, batch_size) -> None:
        """ Class constructor

        Parameters : 
        latent_dim              : size of latent dimension
        batch_size              : size of batch used for training step
        """
        # initialize keras Model class
        super().__init__()
        self.count = 0
        self.latent_dim = latent_dim
        self.generator = self.generate_generator()
        self.discriminator = self.generate_discriminator()
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
    
    def generate_generator(self) -> tf.keras.Sequential:
        """ Function that defines the architecture of the generator
        """
        filters = [13, 128, 264]
        model = tf.keras.Sequential(name="generator")
        model.add(tf.keras.layers.Dense(filters[0], input_dim=self.latent_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Reshape((filters[0], 1)))

        model.add(tf.keras.layers.Conv1DTranspose(filters[1], kernel_size=4, strides=3, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization())

        for filter in filters[2:]:
            model.add(tf.keras.layers.Conv1DTranspose(filter, kernel_size=4, strides=2, padding='same'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(1, kernel_size=4, padding='same', activation='softplus'))
        model.summary()
        return model

    def generate_discriminator(self) -> tf.keras.Model:
        """ Function that defines the architecture of the discriminator
        """
        filters = [64, 264, 128]
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
        """ Function that initializes the loss functions used for the generator and the discriminator

        Parameters : 
        generator_opt              : optimizer used for generator
        discriminator_opt          : optimizer used for discriminator
        disc_loss                  : loss function used for discriminator
        gen_loss                   : loss function used for generator
        """
        super().compile(run_eagerly=True)
        self.g_optimizer = generator_opt
        self.d_optimizer = discriminator_opt
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

    def generate_trajectories(self, size_of_data):
        """ Function that uses the generator to generate new trajectories

        Parameters : 
        size_of_data               : number of samples to generate
        """
        return self.generator(random_generator((size_of_data, self.latent_dim)))          
    
    def train_disc_gen(self, trajectories, Y_, tag):
        """ Function that trains the generator and the discriminator

        Parameters : 
        trajectories               : real molecular trajectories or molecular trajectories generated from generator
        """
        with tf.GradientTape() as tape:
            if tag == "discriminator":
                predictions = self.discriminator(trajectories)
                loss = self.disc_loss(Y_, predictions)
            elif tag == "generator":
                fake_trajectories = self.generator(trajectories)
                # # predictions = self.discriminator(fake_trajectories)
                loss = self.gen_loss(Y_, fake_trajectories)
                # fake_trajectories = self.generator(trajectories)
                # fake_trajectories = tf.cast(fake_trajectories, dtype=tf.float64)
                # combined_samples = tf.concat([fake_trajectories, Y_], axis=0)
                # predictions = self.discriminator(combined_samples)
                # loss = self.gen_loss(predictions, tf.zeros((int(len(predictions)), 1)))
            else:
                raise Exception("Incorrect tag! Should be either discriminator or generator")
        if tag == "discriminator":
            gradiants = tape.gradient(loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(gradiants, self.discriminator.trainable_weights))
        else:
            self.discriminator.trainable = False
            gradiants = tape.gradient(loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradiants, self.generator.trainable_weights))
        return loss            

    def train_step(self, data):
        """ Function that trains the GAN for each batch of data

        Parameters : 
        data                    : real molecular trajectories for each batch
        """
        input_X, energies = data

        # energies = tf.cast(energies, dtype=tf.float64)
        size_of_data_ = int(len(input_X)) # note that this value may change in the last batch
        input_X = tf.cast(input_X, dtype=tf.float64)

        # generating trajectories
        generated_trajectories = self.generate_trajectories(size_of_data_)
        generated_trajectories = tf.cast(generated_trajectories, dtype=tf.float64)
        combined_trajectories = tf.concat([generated_trajectories, input_X], axis=0)

        # labels for differentiating real vs fake trajectories
        combined_label = tf.concat([tf.ones((size_of_data_, 1)), tf.zeros((size_of_data_, 1))], axis=0)

        # Shuffling the inputs randomly
        combined_trajectories, combined_label = random_shuffle(combined_trajectories, combined_label)

        # Training the discriminator
        d_loss = self.train_disc_gen(trajectories=combined_trajectories, Y_=combined_label, tag="discriminator")

        # Generating random labels and concinate with real energies
        random_vector_labels = random_generator((size_of_data_, self.latent_dim))
        # Training the generator
        g_loss = self.train_disc_gen(trajectories=random_vector_labels, Y_=input_X, tag="generator")
        # Monitoring loss
        self.discriminator_loss.update_state(d_loss)
        self.generator_loss.update_state(g_loss)
        return {
            "g_loss": self.generator_loss.result(),
            "d_loss": self.discriminator_loss.result(),
        }
    
def load_weight(weight_path : str):
    """ Function to load the weight of GAN 

    Parameters : 
    weight_path                 : path to the weight of the GAN
    """
    if os.path.isfile(weight_path):
        pass
    else:
        raise ValueError(f"file {weight_path} does not exist")
    return tf.keras.models.load_model(weight_path, custom_objects={"GenAdvNetwork": GenAdvNetwork})

if __name__=="__main__":
    pass