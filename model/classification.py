import tensorflow as tf
from tensorflow import keras


class Classifier(keras.Model):
    def __init__(self, input_dim=(28, 28, 1), name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.model = self.build_classifier()

    def build_classifier(self):
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.input_dim, name='classifier_input'),
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
                tf.keras.layers.Dense(10)
            ],
            name="Classifier"
        )
        return encoder

    def call(self, inputs):
        x = self.model(inputs)
        return x

