import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config


np_config.enable_numpy_behavior()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_latent_space(model, out_dir, n=30, figsize=15, title="Latent space examples", show=False):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 3.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title)
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(os.path.join(out_dir, 'latent_sampled_grid' + '.png'))
    if show:
        plt.show()
    plt.close()


def plot_label_clusters(model, data, labels, out_dir, title="Latent Clustering", show=False):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = model.encoder(data)
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab20', len(class_names))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=cmap)
    ticks = range(len(class_names))
    cbar = plt.colorbar(ticks=ticks)
    cbar.ax.set_yticklabels(class_names)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title)
    plt.savefig(os.path.join(out_dir, 'latent_clusters' + '.png'))
    if show:
        plt.show()
    plt.close()
