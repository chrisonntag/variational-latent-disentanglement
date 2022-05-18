import tensorflow as tf
import numpy as np
from tensorflow import keras
import time


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
    def __init__(self, model, params, optimizer):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.train_stop = Stopper(optimizer)

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
            total_loss = sum(self.model.losses)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return total_loss

    @tf.function
    def val_step(self, x_batch):
        reconstruction = self.model(x_batch, training=False)
        total_loss = sum(self.model.losses)
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
