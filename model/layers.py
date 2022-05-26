import tensorflow as tf
from tensorflow import keras
from model.distributions import log_normal_pdf


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


class MSELoss(keras.layers.Layer):
    """
    Pixel-wise squared difference between y_true and y_pred.
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def call(self, y_true, y_pred):
        r_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
        return 1000 * r_loss


class SigmoidCrossEntropy(keras.layers.Layer):
    def __init__(self):
        super(SigmoidCrossEntropy, self).__init__()

    def call(self, y_true, y_pred):
        # cross entropy between p(x|z) and x for each dim of every datapoint
        # shape=(batch_size, x, y, c)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)

        # shape=(batch_size,), cross entropy for all instances in batch x, logp(x|z)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2, 3]))


class KullbackLeiblerDivergence(keras.layers.Layer):
    def __init__(self):
        super(KullbackLeiblerDivergence, self).__init__()

    def call(self, sample, mean, log_var, analytical=True, prior=log_normal_pdf):
        if analytical:
            kl = 1. + log_var - tf.square(mean) - tf.exp(log_var)
            kl = -0.5 * tf.reduce_sum(kl, axis=-1)
        else:
            # Monte Carlo estimation from a single sample
            logpz = prior(sample, 0., 1.)  # prior, approximated with sampled z
            logqz_x = prior(sample, mean, log_var)  # posterior p(z|x) approximated by learnt q(z|x)

            kl = -(logpz - logqz_x)
        return kl
