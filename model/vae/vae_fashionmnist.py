# VAE Base
import os
from abc import ABC, abstractmethod
import pathlib
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.layers.sampling import Sampling


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class Encoder(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), z_dim=10, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.model = self.build_encoder()
        self.sampling = Sampling()

    def build_encoder(self):
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.input_dim, name='encoder_input'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.z_dim + self.z_dim),
            ],
            name="Encoder"
        )
        encoder.summary()
        return encoder

    def call(self, inputs):
        x = self.model(inputs)

        # generate latent vector Q(z|X) by splitting the last dense layer which is by design 2*z_dim
        z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
        z = self.sampling([z_mean, z_log_var])

        return z_mean, z_log_var, z


class Decoder(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), z_dim=10, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.model = self.build_decoder()

    def build_decoder(self):
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.z_dim,), name='decoder_input'),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ],
            name="Decoder"
        )
        decoder.summary()

        return decoder

    def call(self, inputs):
        return self.model(inputs)


class VariationalAutoEncoderMNIST(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), z_dim=10, beta=0.01):
        super(VariationalAutoEncoderMNIST, self).__init__(name="FashionVAE")
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        self.encoder = Encoder(input_dim, z_dim)  # self.build_encoder()
        self.decoder = Decoder(input_dim, z_dim)  # self.build_decoder()

    def compute_loss_kl_analytically(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return reconstruction_loss + self.beta * kl_loss, reconstruction

    def compute_loss_monte_carlo(self, x, training=False):
        """
        Optimize the single sample Monte Carlo estimate of this expectation:
        log p(x|z) + log p(z) - log q(z|x), where z is sampled from q(z|x)
        """
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=x)
        logpx_z = -tf.reduce_sum(reconstruction_loss, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, z_mean, z_log_var)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x), reconstruction

    def call(self, x):
        total_loss, reconstruction = self.compute_loss_monte_carlo(x)
        self.add_loss(total_loss)

        return reconstruction




