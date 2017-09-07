#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This script contains the MLP class along with associated functions.
"""

from __future__ import print_function

import sys
from math import log10

import numpy as np

from activation import activation_function, activation_derivative, softmax
from loss_function import cross_entropy_loss


def get_iter_batch(X, y, batch_size, iterations):
    """
    Description: A data iterator for batching of image features and labels

    Params:
            X: input array
            y: output array
            batch_size:  size of the batch

    Returns:
            tuple: a batch of image features and labels
    """
    for i in range(iterations):
        ids = np.random.choice(X.shape[0], batch_size)
        batch = (X[ids], y[ids])
        yield batch


class MLP(object):
    def __init__(self, sizes, activation_types, learning_rate=0.02,
                 learning_rate_decay=1., learning_rate_decay_interval=250,
                 l2_reg_lambda=0.001, momentum=0.1, softmax_output=True):
        """
        Description: initializes the biases and weights using a Gaussian
        distribution.
        Biases are not set for 1st layer that is the input layer.

        Params:
                sizes: a list of size L; where L is the number of layers
                    in the deep neural network and each element of list contains
                    the number of neuron in that layer.
                    first and last elements of the list corresponds to the input
                    layer and output layer respectively
                    intermediate layers are hidden layers.
                activation_types: a list of size L-1, of activation fn types
                learning_rate: controls the rate of changes in weights & biases
                l2_reg_lambda: L2 regularization parameter. If no regularization
                    is needed, set this to 0.
                activation_function: Available activation functions are linear,
                    sigmoid, ReLU
                softmax_output: Final layer softmax
        """
        self.num_layers = len(sizes)
        if self.num_layers - 1 != len(activation_types):
            raise ValueError('Activations must only be mentioned for layers other than input layer')
        self.activation_types = activation_types
        self.biases = [np.random.normal(0, 0.08, (y, 1)) for y in sizes[1:]]
        self.weights = [np.random.normal(0, 0.08, (y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_interval = learning_rate_decay_interval
        self.l2_reg_lambda = l2_reg_lambda
        self.momentum = momentum
        self.softmax_output = softmax_output
        self.train_losses = []
        self.test_losses = []

    def feedforward(self, x):
        """
        Description: Forward Passes an image feature matrix through the Deep
        Neural Network Architecture.

        Params:
                x: Image Features

        Returns:
                list: outputs at each layer
                list: activated outputs at each layer
        """
        activation = x.T
        activations = [x.T]
        outs = []
        for b, w, t in zip(self.biases, self.weights, self.activation_types):
            out = np.dot(w, activation) + b
            outs.append(out)
            activation = activation_function(out, type=t)
            activations.append(activation)
        return outs, activations

    def train(self, X, y, batch_size=64, iterations=1000,
              validation=False, validation_data=(), validation_interval=200):
        """Description: Batch-wise trains image features against corresponding
        labels. The weights and biases of the neural network are updated through
        backpropagation on batches using SGD.
        del_b and del_w contain the gradients which are used to update weights
        and biases

        Params:
                X: input array
                y: output array
                batch_size:  size of the batch
                iterations: no. of minibatches to sample and train
                validation: True if validation against test data is needed
                validation_data: tuple of testdata and testlabel matrices
                validation_interval: Interval between iterations at which
                    validation needs to be done
        """
        learning_rate = self.learning_rate
        batch_iter = get_iter_batch(X, y, batch_size, iterations)
        velocity_b = [np.zeros(b.shape) for b in self.biases]
        velocity_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(iterations):
            batch = batch_iter.next()
            del_b = [np.zeros(b.shape) for b in self.biases]
            del_w = [np.zeros(w.shape) for w in self.weights]

            # Obtain weight and bias derivatives from backprop
            loss, delta_del_b, delta_del_w = self.backpropagate(
                batch[0], batch[1], batch_size)
            del_b = [np.add(db, ddb) for db, ddb in zip(del_b, delta_del_b)]
            del_w = [np.add(dw, ddw) for dw, ddw in zip(del_w, delta_del_w)]

            velocity_w = [self.momentum * _vel_w - (learning_rate)
                          * delw for _vel_w, delw in zip(velocity_w, del_w)]
            velocity_b = [self.momentum * _vel_b - (learning_rate)
                          * delb for _vel_b, delb in zip(velocity_b, del_b)]

            # Weight update based on gradient descent
            self.weights = [w + _vel_u for w, _vel_u in zip(self.weights, velocity_w)]
            self.biases = [b + _vel_b for b, _vel_b in zip(self.biases, velocity_b)]
            self.train_losses.append([i, loss])

            line = ("Epoch " + str(i + 1).rjust(int(log10(iterations) + 1))
                    + "/" + str(iterations)
                    + " complete.       Train Loss: %0.10f       " % loss)
            sys.stdout.write('%s\r' % line)
            sys.stdout.flush()

            # Validation at intervals
            if int(i) % validation_interval == 0:
                if validation:
                    t_loss = self.test_loss(validation_data[0], validation_data[1])
                    self.test_losses.append([i, t_loss])
                    line += "Test loss: %0.10f" % t_loss
                    sys.stdout.write('%s\n' % line)
                    sys.stdout.flush()

            # Learning rate scheduling at intervals
            if int(i) % self.learning_rate_decay_interval == 0:
                learning_rate *= self.learning_rate_decay

    def backpropagate(self, x, y, batch_size):
        """
        Description: Based on the derivative (delta) of cost function the
        gradients (rate of change of cost function with respect to weights
        and biases) of weights and biases are calculated.
        cost function here is CrossEntropyLoss cost, hence cost_deriv is :
        delta C = activation(output_layer) - target

        Params:
                X: input array
                y: output array

        Returns:
                float: loss for the current batch prediction
                list: list of layerwise weight derivatives
                list: list of layerwise bias derivatives
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        outs, activations = self.feedforward(x)
        if self.softmax_output:
            y_pred = softmax(outs[-1])
        else:
            y_pred = outs[-1]
        loss = cross_entropy_loss(y_pred.T, y)

        delta_cost = y_pred - y.T
        delta = delta_cost

        # Gradients for last layer
        del_b[-1] = np.mean(delta, axis=1).reshape(-1, 1)
        del_w[-1] = np.dot(delta, activations[-2].T)

        # Gradients for layers other than last layer
        for l in xrange(2, self.num_layers):
            out = outs[-l]
            delta_activation = activation_derivative(out, name=self.activation_types[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * delta_activation
            del_b[-l] = np.mean(delta, axis=1).reshape(-1, 1)
            del_w[-l] = (np.dot(delta, activations[-l - 1].T) + self.l2_reg_lambda * (self.weights[-l])) / batch_size
        return loss, del_b, del_w

    def eval(self, X, y):
        """
        Description: Based on trained(updated) weights and biases, predict output labels,
        compare them with original outputs and calculate accuracy

        Params:
                X: input array
                y: output array
        """
        outs, activations = self.feedforward(X)
        if self.softmax_output == True:
            y_pred = softmax(outs[-1])
        else:
            y_pred = outs[-1]
        preds = np.argmax(y_pred, axis=0)
        actuals = np.argmax(y, axis=1)
        count = np.sum(preds == actuals)
        print("\nAccuracy: %f" % ((float(count) / X.shape[0]) * 100))

    def test_loss(self, X, y):
        """
        Description: Based on trained(updated) weights and biases, predict output vectors and
        evaluate the cross-entropy loss

        Params:
                X: input array
                y: output array

        Returns:
                float: Test loss
        """
        outs, activations = self.feedforward(X)
        if self.softmax_output == True:
            y_pred = softmax(outs[-1])
        else:
            y_pred = outs[-1]
        loss = cross_entropy_loss(y_pred.T, y)
        return loss

    def predict(self, X, num_predictions=1):
        """
        Description: Based on trained(updated) weights and biases, generates
        best predictions for the input

        Params:
                X: input array
                num_predictions: Number of predictions to make

        Returns:
                ndarray: Array of predictions
        """
        outs, activations = self.feedforward(X)
        if self.softmax_output:
            y_pred = softmax(outs[-1])
        else:
            y_pred = outs[-1]
        preds = np.argpartition(y_pred, -1 * num_predictions, axis=0)[-1 * num_predictions:][::-1]
        return preds.T

    def loss_stats(self):
        """
        Description: Returns the evolution of train data and test data losses.

        Returns:
                ndarray: train losses at each iteration
                ndarray: test losses at specified intervals
        """
        return np.array(self.train_losses), np.array(self.test_losses)
