from util.experiment import Experiment
from tensorflow import keras
import tensorflow as tf
import numpy as np
from util.plotting import *


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0

experiment = Experiment(name="08-05-2022_2240_lead-talk-right-line", base_path="experiments/")
model = experiment.load_model()

plot_latent_space(model, experiment.vis_dir)
plot_label_clusters(model, train_images, train_labels, experiment.vis_dir)
