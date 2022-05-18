import tensorflow as tf
from tensorflow import keras
from model.layers.sampling import Sampling
from model.distributions import log_normal_pdf


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

        # Sample latent vector Q(z|X) by splitting the last dense layer which is by design 2*z_dim
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
    def __init__(self, input_dim=(28, 28, 1), z_dim=10, beta=2, prior=log_normal_pdf):
        super(VariationalAutoEncoderMNIST, self).__init__(name="FashionVAE")
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        self.prior = prior
        self.encoder = Encoder(input_dim, z_dim)
        self.decoder = Decoder(input_dim, z_dim)

    def compute_loss(self, x, analytic=True):
        """
        Optimize the single sample Monte Carlo estimate of this expectation:
        log p(x|z) + log p(z) - log q(z|x), where z is sampled from q(z|x)
        """
        z_mean, z_log_var, z = self.encoder(x)  # samples z based on distribution of x: q(z|x) posterior distribution
        reconstruction = self.decoder(z)  # p(x|z)

        # Reconstruction loss
        # shape=(batch_size, 28, 28, 1), cross entropy between p(x|z) and x for each dim of every datapoint
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=x)
        # shape=(batch_size,), cross entropy for all instances in batch x, logp(x|z)
        logpx_z = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2, 3]))

        # KL Divergence logp(z) - logq(z|x), if we assume q to be a multivariate Gaussian distribution
        if analytic:
            kl_divergence = - 0.5 * tf.reduce_sum(
                1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)  # shape=(batch_size, )
            kl_divergence = tf.reduce_mean(kl_divergence)
        else:
            # Monte Carlo estimation from a single sample
            logpz = self.prior(z, 0., 1.)  # prior, approximated with sampled z
            logqz_x = self.prior(z, z_mean, z_log_var)  # posterior p(z|x) approximated by learnt q(z|x)

            kl_divergence = logpz - logqz_x

        # -mean( logp(x|z) + logp(z) - logq(z|x) )
        elbo = logpx_z + self.beta * kl_divergence
        return -elbo, reconstruction

    def call(self, x):
        total_loss, reconstruction = self.compute_loss(x, analytic=True)
        self.add_loss(total_loss)

        return reconstruction




