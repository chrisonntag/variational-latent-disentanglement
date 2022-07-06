import tensorflow as tf
from tensorflow import keras


class Encoder(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), label_dim=10, z_dim=10, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.z_dim = z_dim
        self.model = Encoder.build_encoder(self.input_dim, self.label_dim, self.z_dim)

    @staticmethod
    def build_latent_layer(inputs, z_dim):
        x = tf.keras.layers.Dense(z_dim + z_dim)(inputs)

        return x

    @staticmethod
    def build_main_encoder(inputs):
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation=None)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)

        return x

    @staticmethod
    def build_encoder(input_shape, label_shape, z_dim):
        cond_inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2] + label_shape), name='encoder_input')

        main_encoder = Encoder.build_main_encoder(cond_inputs)
        latent = Encoder.build_latent_layer(main_encoder, z_dim)

        model = keras.Model(
            inputs=cond_inputs,
            outputs=latent
        )

        return model

    def call(self, inputs):
        latent = self.model(inputs)

        # Sample latent vector Q(z|X) by splitting the last dense layer which is by design 2*z_dim
        z_mean, z_log_var = tf.split(latent, num_or_size_splits=2, axis=1)

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


class ConditionalVAE(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), label_dim=10, z_dim=10, beta=2):
        super(ConditionalVAE, self).__init__(name="FashionVAE")
        self.z = None
        self.z_log_var = None
        self.z_mean = None
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = beta
        self.label_dim = label_dim
        self.encoder = Encoder(input_dim, label_dim, z_dim)
        self.decoder = Decoder(input_dim, z_dim)

    @classmethod
    def from_saved_model(cls, model, params):
        instance = cls(input_dim=params['input_dim'], label_dim=params['label_dim'], z_dim=params['z_dim'], beta=params['beta'])
        instance.encoder = model.encoder
        instance.decoder = model.decoder
        return instance

    def concat_image_label(self, inputs):
        x = inputs[0]
        y = inputs[1]
        reshaped_labels = tf.reshape(y, [-1, 1, 1,
                                         self.label_dim])  # reshape to picture size (batch, width, height, channels)
        ones = tf.ones([x.shape[0]] + [self.input_dim[0], self.input_dim[1]] + [self.label_dim])  # fill with ones
        reshaped_labels = ones * reshaped_labels
        concat_inputs = tf.concat([x, reshaped_labels], axis=3)

        return concat_inputs

    def encode(self, xy):
        z_mean, z_log_var = self.encoder(xy)
        return z_mean, z_log_var

    def encode_dist(self, xy):
        z_mean, z_log_var, _ = self.encode(xy)
        return keras.backend.random_normal(
            mean=z_mean, stddev=z_log_var, shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))

    def reparameterize(self, z_mean, z_log_var):
        # samples z based on distribution (z_mean, z_log_var) of x: q(z|x) posterior distribution
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = keras.backend.random_normal(shape=(batch, dim))

        # transfor z_log_var to stddev
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    def call(self, inputs):
        self.z_mean, self.z_log_var = self.encode(inputs)
        self.z = self.reparameterize(self.z_mean, self.z_log_var)
        outputs = self.decode(self.z)

        return outputs

