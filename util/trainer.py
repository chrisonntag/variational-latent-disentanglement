import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from model.distributions import log_normal_pdf
from model.layers import KullbackLeiblerDivergence, MSELoss


class Stopper:
    def __init__(self, optimizer, min_delta=0.01, patience=4, max_lr_decay=10, lr_decay_factor=0.3):
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


class Trainer:
    def __init__(self, model, params, optimizer, prior=log_normal_pdf):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.train_stop = Stopper(optimizer)

        self.prior = prior

        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
        ]

    def compute_loss(self, x, analytic=True, training=True):
        """
        Optimize the single sample Monte Carlo estimate of this expectation:
        log p(x|z) + log p(z) - log q(z|x), where z is sampled from q(z|x)
        """
        z_mean, z_log_var, z, reconstruction = self.model(x, training=training)

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
        elbo = logpx_z + self.model.beta * kl_divergence
        return -elbo, reconstruction

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            x_reconstruction = self.model(x_batch)

            rec_los = tf.math.reduce_mean(MSELoss()(x_batch, x_reconstruction))
            kl_loss = tf.math.reduce_mean(KullbackLeiblerDivergence()(self.model.z_mean, self.model.z_log_var))

            total_loss = rec_los + self.model.beta * kl_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return total_loss

    @tf.function
    def val_step(self, x_batch):
        x_reconstruction = self.model(x_batch)

        rec_los = tf.math.reduce_mean(MSELoss()(x_batch, x_reconstruction))
        kl_loss = tf.math.reduce_mean(KullbackLeiblerDivergence()(self.model.z_mean, self.model.z_log_var))

        total_loss = rec_los + self.model.beta * kl_loss
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
            self.train_loss_tracker.reset_state()  # Reset training metrics at the end of each epoch

            # Validation Dataset
            for step, x_batch_val in enumerate(val_ds):
                loss = self.val_step(x_batch_val)
                self.val_loss_tracker.update_state(loss)
                if step % 100 == 0:
                    print("val step %d: mean loss = %.4f" % (step, self.val_loss_tracker.result()))

            val_loss.append(self.val_loss_tracker.result())
            self.val_loss_tracker.reset_state()

            print("Time taken: %.2fs" % (time.time() - start_time))
            if self.train_stop.check_stop_training(val_loss):
                break

        return train_loss, val_loss
