#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper function containing activation functions.
"""

import numpy as np


def cross_entropy_loss(a, y):
    """Description: Calculates the average cross entropy loss for the given predicted
    vectors with respect to the actual y vectors

    Params:
        a: Array of predicted vectors
        y: Array of actual vectors

    Returns:
        float32: Cross entropy loss for the current batch
    """
    return np.sum(np.nan_to_num(np.multiply(-y, np.log(a)))) / a.shape[0]
