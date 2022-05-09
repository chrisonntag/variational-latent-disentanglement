import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.vae.vae_fashionmnist import VariationalAutoEncoderMNIST
from util.experiment import Experiment
from util.trainer import Trainer

# Download Dataset
fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255.0
valid_images = valid_images.astype('float32') / 255.0

# Add third single dimension
train_images = tf.expand_dims(train_images, axis=-1)
valid_images = tf.expand_dims(valid_images, axis=-1)

# Setup training
betas = [0.01, 0.1, 0.5, 1, 1.2, 1.5, 2]
dims = [1, 2, 3, 4, 8, 12, 16]

params_list = []
for b in betas:
    for dim in dims:
        params_list.append(
            {
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "epochs": 40,
                "batch_size": 128,
                "latent_dim": dim,
                "beta": b,
            }
        )

for params in params_list:
    optimizers = {"Adam": keras.optimizers.Adam(learning_rate=params['learning_rate'])}
    optimizer = optimizers[params['optimizer']]

    # Create dataset - only needed for manual training
    train_ds = (tf.data.Dataset.from_tensor_slices(train_images)).shuffle(buffer_size=1024).batch(params['batch_size'])
    valid_ds = (tf.data.Dataset.from_tensor_slices(valid_images)).batch(params['batch_size'])

    # Create model and do Training
    vae = VariationalAutoEncoderMNIST(z_dim=params['latent_dim'], beta=params['beta'])
    trainer = Trainer(model=vae, params=params, optimizer=optimizer)

    train_history, val_history = trainer.train(train_ds, valid_ds)
    history = {"train_loss": train_history, "val_loss": val_history}

    experiment = Experiment(base_path="experiments/")
    experiment.save(params, model=vae, history=history)
    experiment.plot(history['train_loss'], history['val_loss'])


