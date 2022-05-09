import os
from util.experiment import Experiment
from tensorflow import keras
import tensorflow as tf
import numpy as np
from util.plotting import *


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0

base_path = 'experiments'
experiments = {}

# Load all experiments with basic configuration into a dictionary
for directory in os.listdir(base_path):
    d = os.path.join(base_path, directory)
    if os.path.isdir(d):
        experiment = Experiment(name=directory, base_path=base_path)

        if experiment.params['latent_dim'] == 2:
            model = experiment.load_model()

            plot_latent_space(model, experiment.vis_dir, title="beta %.2f" % experiment.params['beta'])
            plot_label_clusters(model, train_images, train_labels, experiment.vis_dir, title="beta %.2f" % experiment.params['beta'])

        # Save to dict
        experiments[directory] = {
            'params': experiment.params,
            'experiment': experiment
        }

