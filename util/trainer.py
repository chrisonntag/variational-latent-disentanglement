import tensorflow as tf
from tensorflow import keras
import time


class Trainer:
    def __init__(self, model, params, optimizer):
        self.model = model
        self.params = params
        self.optimizer = optimizer

        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
        ]

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            reconstruction = self.model(x_batch, training=True)

            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x_batch, reconstruction), axis=(1, 2)
                )
            )
            # Add KLD regularization loss
            total_loss = reconstruction_loss + sum(self.model.losses)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return total_loss

    @tf.function
    def val_step(self, x_batch):
        reconstruction = self.model(x_batch, training=False)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x_batch, reconstruction), axis=(1, 2)
            )
        )
        # Add KLD regularization loss
        total_loss = reconstruction_loss + sum(self.model.losses)
        return total_loss

    def train(self, train_ds, val_ds):
        train_loss = []
        val_loss = []
        for epoch in range(self.params['epochs']):
            print("Epoch %d/%d" % (epoch+1, self.params['epochs']))
            start_time = time.time()

            # Train Dataset
            for step, x_batch_train in enumerate(train_ds):
                loss = self.train_step(x_batch_train)
                self.train_loss_tracker.update_state(loss)
                if step % 100 == 0:
                    print("step %d: mean loss = %.4f" % (step, self.train_loss_tracker.result()))

            train_loss.append(self.train_loss_tracker.result())
            self.train_loss_tracker.reset_states()  # Reset training metrics at the end of each epoch

            # Validation Dataset
            for step, x_batch_val in enumerate(val_ds):
                loss = self.val_step(x_batch_val)
                self.val_loss_tracker.update_state(loss)
                if step % 100 == 0:
                    print("val step %d: mean loss = %.4f" % (step, self.val_loss_tracker.result()))

            val_loss.append(self.val_loss_tracker.result())
            self.val_loss_tracker.reset_states()

            print("Time taken: %.2fs" % (time.time() - start_time))

        return train_loss, val_loss
