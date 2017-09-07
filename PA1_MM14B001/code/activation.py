#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper file containing activation functions
"""

import numpy as np


def sigmoid(x):
    """Description: Calculates the sigmoid for each value in the the input array
    Params:
        x: Array for which sigmoid is to be calculated

    Returns:
        ndarray: Sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-x))


def delta_sigmoid(x):
    """Description: Calculates the sigmoid derivative for the input array
    Params:
        x: Array for which sigmoid derivative is to be calculated

    Returns:
        ndarray: Sigmoid derivative of the input
    """
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    """Description: Calculates softmax for each set of scores in the input array

    Params:
        x: Array for which softmax is to be calculated
            (axis_0 is the feature dimension, axis_1 is the n_samples dim)

    Returns:
        ndarray: Softmax of the input
    """
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


def relu(x):
    """Description: Calculates ReLU for each value in the input array

    Params:
        x: Array for which ReLU is to be calculated

    Returns:
        ndarray: ReLU of the input
    """
    return np.maximum(x, 0)


def delta_relu(x):
    """
	Description: Calculates the ReLU derivative for the input array

	Params:
			x: Array for which ReLU derivative is to be calculated

	Returns:
			ndarray: ReLU derivative of the input
	"""
    return np.greater(x, 0).astype(np.float32)


def linear(x):
    """
	Description: Calculates the linear activation for the input array

	Params:
			x: Array for which linear activation is to be calculated

	Returns:
			ndarray: Linear activation of the input
	"""
    return x


def delta_linear(x):
    """
	Description: Calculates the linear activation derivative for for the input array

	Params:
			x: Array for which linear activation derivative is to be calculated

	Returns:
			ndarray: Linear activation derivative of the input
	"""
    return np.ones(x.shape).astype(np.float32)


def activation_function(x, type="linear"):
    """
	Description: Helper function for calculating activation of the input

	Params:
			out: Array for which activation is to be calculated
			type: Type of the activation function 
			(can be linear, sigmoid, relu, softmax)

	Returns:
			ndarray: Activation of the input
	"""
    if (type == "linear"):
        return linear(x)
    elif type == "sigmoid":
        return sigmoid(x)
    elif type == "relu":
        return relu(x)
    elif type == "softmax":
        return softmax(x)
    else:
        raise ValueError('Invalid activation type entered')


def activation_derivative(x, name="linear"):
    """Description: Helper function for calculating activation derivative of the input

	Params:
			out: Array for which activation derivative is to be calculated
			name: Type of the activation derivative function
			(can be linear, sigmoid, relu, softmax)

	Returns:
			ndarray: Activation derivative of the input
	"""
    if (name == "linear"):
        return delta_linear(x)
    elif (name == "sigmoid"):
        return delta_sigmoid(x)
    elif (name == "relu"):
        return delta_relu(x)
    else:
        raise ValueError('Invalid activation type entered')
