import os
from util.experiment import Experiment
from tensorflow import keras
import tensorflow as tf
import numpy as np
from util.plotting import *
from util.experiment import load_experiments

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0

exps = load_experiments(with_params={"latent_dim": 2})
for key in exps.keys():
    experiment = exps[key]['experiment']
    model = experiment.load_model()
    plot_latent_space(model, experiment.vis_dir, title="beta %.2f" % experiment.params['beta'])
    plot_label_clusters(model, train_images, train_labels, experiment.vis_dir,
                        title="beta %.2f" % experiment.params['beta'])
