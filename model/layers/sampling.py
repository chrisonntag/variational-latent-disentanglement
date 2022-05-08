import tensorflow as tf
from tensorflow import keras


class Sampling(keras.layers.Layer):
    """
    Custom sampling layer for latent space
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Reparameterize
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super(Sampling, self).get_config()
