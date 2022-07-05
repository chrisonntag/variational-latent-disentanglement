import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from model.distributions import log_normal_pdf
import model.layers as layers


class Stopper:
    """
    Early Stopper class offers callbacks that can be used during training in order to decide whether
    stopping the learning process early is beneficial

    Source: https://github.com/kingusiu/vande/blob/master/training.py
    """
    def __init__(self, optimizer, min_delta=0.001, patience=4, max_lr_decay=10, lr_decay_factor=0.2):
        self.optimizer = optimizer
        self.min_delta = min_delta
        self.patience = patience
        self.patience_curr = 0
        self.max_lr_decay = max_lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_n = 0

    def callback_early_stopping(self, loss_list, min_delta=0.1, patience=4):
        # increase loss-window width at each epoch
        self.patience_curr += 1
        if self.patience_curr < patience:
            return False
        # compute difference of the last #patience epoch losses
        mean = np.mean(loss_list[-patience:])
        deltas = np.absolute(np.diff(loss_list[-patience:]))
        # return true if all relative deltas are smaller than min_delta
        return np.all((deltas / mean) < min_delta)

    def check_stop_training(self, losses):
        if self.callback_early_stopping(losses, min_delta=self.min_delta, patience=self.patience):
            print('-'*7 + ' Early stopping for last ' + str(self.patience) + ' validation losses ' +
                  str([l.numpy() for l in losses[-self.patience:]]) + '-'*7)
            if self.lr_decay_n >= self.max_lr_decay:
                # stop the training
                return True
            else:
                # decrease the learning rate
                curr_lr = self.optimizer.learning_rate.numpy()
                self.optimizer.learning_rate.assign(curr_lr * self.lr_decay_factor)
                self.lr_decay_n += 1
                # reset patience window each time lr has been decreased
                self.patience_curr = 0
                print('decreasing learning rate from {:.3e} to {:.3e}'.format(curr_lr, self.optimizer.learning_rate.numpy()))

        return False


class BCVAETrainer:
    def __init__(self, model, params, optimizer, prior=log_normal_pdf):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.train_stop = Stopper(optimizer)

        self.prior = prior

        self.train_class_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.train_rec_loss_tracker = keras.metrics.Mean(name="train_rec_loss")
        self.train_kl_loss_tracker = keras.metrics.Mean(name="train_kl_loss")
        self.val_class_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_rec_loss_tracker = keras.metrics.Mean(name="val_rec_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.train_class_loss_tracker,
            self.train_rec_loss_tracker,
            self.train_kl_loss_tracker,
            self.val_class_loss_tracker,
            self.val_rec_loss_tracker,
            self.val_kl_loss_tracker
        ]

    @tf.function
    def train_step(self, batch):
        x_batch = batch[0]
        y_batch = batch[1]

        with tf.GradientTape() as tape:
            outputs = self.model(x_batch)
            x_reconstruction = outputs[0]
            y_pred = outputs[1]

            # rec_loss = keras.losses.MeanSquaredError()(x_batch, x_reconstruction)
            rec_loss = layers.MSELoss()(x_batch, x_reconstruction)
            kl_loss = layers.KullbackLeiblerDivergence()(
                self.model.z, self.model.z_mean, self.model.z_log_var, analytical=True, prior=self.prior
            )
            total_loss = rec_loss + self.model.beta * kl_loss

            # CategoricalCrossentropy loss
            class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            class_loss = class_loss_fn(y_batch, y_pred)

            total_loss += class_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return class_loss, rec_loss, kl_loss

    @tf.function
    def val_step(self, batch):
        x_batch = batch[0]
        y_batch = batch[1]

        outputs = self.model(x_batch, training=False)
        x_reconstruction = outputs[0]
        y_pred = outputs[1]

        rec_loss = layers.MSELoss()(x_batch, x_reconstruction)
        kl_loss = layers.KullbackLeiblerDivergence()(
            self.model.z, self.model.z_mean, self.model.z_log_var, analytical=True, prior=self.prior
        )
        total_loss = rec_loss + self.model.beta * kl_loss

        class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        class_loss = class_loss_fn(y_batch, y_pred)

        total_loss += class_loss

        return class_loss, rec_loss, kl_loss

    def train(self, train_ds, val_ds):
        train_class_loss = []
        train_rec_loss = []
        train_kl_loss = []
        val_class_loss = []
        val_rec_loss = []
        val_kl_loss = []

        for epoch in range(self.params['epochs']):
            print("Epoch %d/%d" % (epoch+1, self.params['epochs']))
            start_time = time.time()

            # Train Dataset
            for step, batch_train in enumerate(train_ds):
                class_loss, rec_loss, kl_loss = self.train_step(batch_train)
                self.train_class_loss_tracker.update_state(class_loss)
                self.train_rec_loss_tracker.update_state(rec_loss)
                self.train_kl_loss_tracker.update_state(kl_loss)
                if step % 100 == 0:
                    print("step %d: mean class loss = %.4f, rec = %.4f, kl = %.4f" % (
                        step,
                        self.train_class_loss_tracker.result(),
                        self.train_rec_loss_tracker.result(),
                        self.train_kl_loss_tracker.result()
                    ))

            train_class_loss.append(self.train_class_loss_tracker.result())
            train_rec_loss.append(self.train_rec_loss_tracker.result())
            train_kl_loss.append(self.train_kl_loss_tracker.result())

            # Reset training metrics at the end of each epoch
            self.train_class_loss_tracker.reset_state()
            self.train_rec_loss_tracker.reset_state()
            self.train_kl_loss_tracker.reset_state()

            # Validation Dataset
            for step, batch_val in enumerate(val_ds):
                class_loss, rec_loss, kl_loss = self.val_step(batch_val)
                self.val_class_loss_tracker.update_state(class_loss)
                self.val_rec_loss_tracker.update_state(rec_loss)
                self.val_kl_loss_tracker.update_state(kl_loss)
                if step % 100 == 0:
                    print("val step %d: mean class loss = %.4f, rec = %.4f, kl = %.4f" % (
                        step,
                        self.val_class_loss_tracker.result(),
                        self.val_rec_loss_tracker.result(),
                        self.val_kl_loss_tracker.result()
                    ))

            val_class_loss.append(self.val_class_loss_tracker.result())
            val_rec_loss.append(self.val_rec_loss_tracker.result())
            val_kl_loss.append(self.val_kl_loss_tracker.result())

            # Reset training metrics at the end of each epoch
            self.val_class_loss_tracker.reset_state()
            self.val_rec_loss_tracker.reset_state()
            self.val_kl_loss_tracker.reset_state()

            print("Time taken: %.2fs" % (time.time() - start_time))
            if self.train_stop.check_stop_training(val_class_loss):
                break

        history = {
            "class_loss": train_class_loss,
            "train_rec_loss": train_rec_loss,
            "train_kl_loss": train_kl_loss,
            "val_class_loss": val_class_loss,
            "val_rec_loss": val_rec_loss,
            "val_kl_loss": val_kl_loss
        }

        return history


class CVAETrainer:
    def __init__(self, model, params, optimizer, prior=log_normal_pdf):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.train_stop = Stopper(optimizer)

        self.prior = prior

        self.train_rec_loss_tracker = keras.metrics.Mean(name="train_rec_loss")
        self.train_kl_loss_tracker = keras.metrics.Mean(name="train_kl_loss")
        self.val_rec_loss_tracker = keras.metrics.Mean(name="val_rec_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.train_rec_loss_tracker,
            self.train_kl_loss_tracker,
            self.val_rec_loss_tracker,
            self.val_kl_loss_tracker
        ]

    @tf.function
    def train_step(self, batch):
        x_batch = batch[0]
        y_batch = batch[1]

        with tf.GradientTape() as tape:
            outputs = self.model(x_batch, y_batch)
            x_reconstruction = outputs

            # rec_loss = keras.losses.MeanSquaredError()(x_batch, x_reconstruction)
            rec_loss = layers.MSELoss()(x_batch, x_reconstruction)
            kl_loss = layers.KullbackLeiblerDivergence()(
                self.model.z, self.model.z_mean, self.model.z_log_var, analytical=True, prior=self.prior
            )
            total_loss = rec_loss + self.model.beta * kl_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return rec_loss, kl_loss

    @tf.function
    def val_step(self, batch):
        x_batch = batch[0]
        y_batch = batch[1]

        outputs = self.model(x_batch, y_batch, training=False)
        x_reconstruction = outputs

        rec_loss = layers.MSELoss()(x_batch, x_reconstruction)
        kl_loss = layers.KullbackLeiblerDivergence()(
            self.model.z, self.model.z_mean, self.model.z_log_var, analytical=True, prior=self.prior
        )
        total_loss = rec_loss + self.model.beta * kl_loss
        return rec_loss, kl_loss

    def train(self, train_ds, val_ds):
        train_rec_loss = []
        train_kl_loss = []
        val_rec_loss = []
        val_kl_loss = []

        for epoch in range(self.params['epochs']):
            print("Epoch %d/%d" % (epoch+1, self.params['epochs']))
            start_time = time.time()

            # Train Dataset
            for step, batch_train in enumerate(train_ds):
                rec_loss, kl_loss = self.train_step(batch_train)
                self.train_rec_loss_tracker.update_state(rec_loss)
                self.train_kl_loss_tracker.update_state(kl_loss)
                if step % 100 == 0:
                    print("step %d: rec = %.4f, kl = %.4f" % (
                        step,
                        self.train_rec_loss_tracker.result(),
                        self.train_kl_loss_tracker.result()
                    ))

            train_rec_loss.append(self.train_rec_loss_tracker.result())
            train_kl_loss.append(self.train_kl_loss_tracker.result())

            # Reset training metrics at the end of each epoch
            self.train_rec_loss_tracker.reset_state()
            self.train_kl_loss_tracker.reset_state()

            # Validation Dataset
            for step, batch_val in enumerate(val_ds):
                rec_loss, kl_loss = self.val_step(batch_val)
                self.val_rec_loss_tracker.update_state(rec_loss)
                self.val_kl_loss_tracker.update_state(kl_loss)
                if step % 100 == 0:
                    print("val step %d: rec = %.4f, kl = %.4f" % (
                        step,
                        self.val_rec_loss_tracker.result(),
                        self.val_kl_loss_tracker.result()
                    ))

            val_rec_loss.append(self.val_rec_loss_tracker.result())
            val_kl_loss.append(self.val_kl_loss_tracker.result())

            # Reset training metrics at the end of each epoch
            self.val_rec_loss_tracker.reset_state()
            self.val_kl_loss_tracker.reset_state()

            print("Time taken: %.2fs" % (time.time() - start_time))
            if self.train_stop.check_stop_training(val_kl_loss):
                break

        history = {
            "train_rec_loss": train_rec_loss,
            "train_kl_loss": train_kl_loss,
            "val_rec_loss": val_rec_loss,
            "val_kl_loss": val_kl_loss
        }

        return history
