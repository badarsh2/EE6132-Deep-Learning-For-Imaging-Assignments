#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This file contains various utility functions such as I/O utilities, encoding etc.
"""

import struct

import matplotlib.pyplot as plt
import numpy as np


def read_idx(filename):
    """Description: Parses MNIST binary files and returns labels / feature matrices of images

    Params:
        filename: Path to the binary file

    Returns:
        ndarray: 2D / 3D Matrix data
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def indices_to_one_hot(data, nb_classes):
    """Description: Converts iterable of indices to one-hot encoded arrays

    Params:
        data: List / Array of labels
        nb_classes: Number of classes in the distribution

    Returns:
        ndarray: One-hot vectors
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def generate_image_outputs(X_samples, pred_samples, path="."):
    """Description: Generates 28 x 28 2D images from given matrix, displays the
    top predictions for the image and saves them to the specified location

    Params:
        X_samples: Image feature matrix of shape (n_images, 784)
        pred_samples: Predictions for the images; of shape (n_images, n_predictions)
        path: Path to the folder where the image needs to be saved
    """
    for i in range(X_samples.shape[0]):
        data = X_samples[i].reshape(28, 28)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, interpolation='nearest', cmap="gray")
        plt.text(0.5, 0.05, 'Predictions: ' + str(pred_samples[i]), fontsize=20,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, color='white')
        plt.savefig(path + "/sample_" + str(i + 1) + ".png")
        plt.close()


def plot_learning_curve(X, y, colors, labels, title="", path="."):
    """Description: Plots a learning curve for the given data

    Params:
            X: List of data for the X axis
            y: List of data for the Y axis
            colors: List of colors for the data
            labels: List of labels for the data
            path: Path to the folder where the plot needs to be saved
    """
    for _x, _y, _color, _label in zip(X, y, colors, labels):
        plt.plot(_x, _y, _color, label=_label)
    plt.suptitle(title)
    plt.xlabel('Iterations')
    plt.ylabel('Cross entropy loss')
    plt.legend()
    plt.gca().set_ylim([0, 5])
    plt.savefig(path)
    plt.close()
