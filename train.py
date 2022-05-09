import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.vae.vae_fashionmnist import VariationalAutoEncoderMNIST
from util.experiment import Experiment
from util.trainer import Trainer
from model.distributions import log_normal_pdf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


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
betas = [2, 4, 8, 10]
dims = [2, 3, 8, 12]

params_list = []
for b in betas:
    for dim in dims:
        # must be primitive datatypes, because they are stored as pickles and used
        # for filtering in the visualization process.
        params_list.append(
            {
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "epochs": 20,
                "batch_size": 400,
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
    vae = VariationalAutoEncoderMNIST(z_dim=params['latent_dim'], beta=params['beta'], prior=log_normal_pdf)
    trainer = Trainer(model=vae, params=params, optimizer=optimizer)

    train_history, val_history = trainer.train(train_ds, valid_ds)
    history = {"train_loss": train_history, "val_loss": val_history}

    experiment = Experiment(base_path="experiments/")
    experiment.save(params, model=vae, history=history)
    experiment.plot(history['train_loss'], history['val_loss'])


