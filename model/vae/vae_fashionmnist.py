import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from model.layers import Sampling


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
                    filters=32, kernel_size=3, strides=(2, 2), activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.z_dim + self.z_dim),
            ],
            name="Encoder"
        )
        return encoder

    def call(self, inputs):
        x = self.model(inputs)

        # Sample latent vector Q(z|X) by splitting the last dense layer which is by design 2*z_dim
        z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)

        return z_mean, z_log_var


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
                    activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ],
            name="Decoder"
        )

        return decoder

    def call(self, inputs):
        return self.model(inputs)


class VariationalAutoEncoderMNIST(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), z_dim=10, beta=2):
        super(VariationalAutoEncoderMNIST, self).__init__(name="FashionVAE")
        self.z = None
        self.z_log_var = None
        self.z_mean = None
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        self.encoder = Encoder(input_dim, z_dim)
        self.decoder = Decoder(input_dim, z_dim)

    @classmethod
    def from_saved_model(cls, model, params):
        instance = cls(input_dim=params['input_dim'], z_dim=params['z_dim'], beta=params['beta'])
        instance.encoder = model.encoder
        instance.decoder = model.decoder
        return instance

    def encode(self, x):
        z_mean, z_log_var = self.encoder(x)
        return z_mean, z_log_var

    def encode_dist(self, x):
        z_mean, z_log_var = self.encoder(x)
        return tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_log_var)

    def reparameterize(self, z_mean, z_log_var):
        # samples z based on distribution (z_mean, z_log_var) of x: q(z|x) posterior distribution
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    def call(self, inputs):
        self.z_mean, self.z_log_var = self.encode(inputs)
        self.z = self.reparameterize(self.z_mean, self.z_log_var)
        outputs = self.decode(self.z)

        return outputs



