#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Driver program to train a MLP on MNIST dataset.
"""

import numpy as np

from utils import read_idx, indices_to_one_hot, generate_image_outputs, plot_learning_curve
from neural_network import MLP

# Paths for MNIST dataset
DATA_TRAIN = "../data/train-images.idx3-ubyte"
DATA_TEST = "../data/t10k-images.idx3-ubyte"
LABELS_TRAIN = "../data/train-labels.idx1-ubyte"
LABELS_TEST = "../data/t10k-labels.idx1-ubyte"


if __name__ == "__main__":
    # Reading binaries and creating I&O arrays
    X_train = read_idx(DATA_TRAIN).astype(np.float32)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

    X_test = read_idx(DATA_TEST).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    y_train = read_idx(LABELS_TRAIN)
    y_train = indices_to_one_hot(y_train, 10).reshape(-1, 10)

    y_test = read_idx(LABELS_TEST)
    y_test = indices_to_one_hot(y_test, 10).reshape(-1, 10)

    # Normalization
    X_train /= 255.0
    X_test /= 255.0

    print("------------------------------------------------------------------------")
    print("MLP with Sigmoid activation; learning rate=1e-2; No scheduling")
    print("------------------------------------------------------------------------")
    model_1 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["sigmoid", "sigmoid", "sigmoid", "linear"],
                  learning_rate=1e-2, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_1.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_1.eval(X_test, y_test)
    train_loss_sigmoid_1, test_loss_sigmoid_1 = model_1.loss_stats()
    plot_learning_curve([train_loss_sigmoid_1[:, 0], test_loss_sigmoid_1[:, 0]],
                        [train_loss_sigmoid_1[:, 1], test_loss_sigmoid_1[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-2",
                        path="../outputs/train_test_loss_sigmoid_1e-2.png")

    print("------------------------------------------------------------------------")
    print("MLP with Sigmoid activation; learning rate=1e-3; No scheduling")
    print("------------------------------------------------------------------------")
    model_2 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["sigmoid", "sigmoid", "sigmoid", "linear"],
                  learning_rate=1e-3, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_2.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_2.eval(X_test, y_test)
    train_loss_sigmoid_2, test_loss_sigmoid_2 = model_2.loss_stats()
    plot_learning_curve([train_loss_sigmoid_2[:, 0], test_loss_sigmoid_2[:, 0]],
                        [train_loss_sigmoid_2[:, 1], test_loss_sigmoid_2[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-3",
                        path="../outputs/train_test_loss_sigmoid_1e-3.png")

    print("------------------------------------------------------------------------")
    print("MLP with Sigmoid activation; learning rate=1e-4; No scheduling")
    print("------------------------------------------------------------------------")
    model_3 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["sigmoid", "sigmoid", "sigmoid", "linear"],
                  learning_rate=1e-4, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_3.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_3.eval(X_test, y_test)
    train_loss_sigmoid_3, test_loss_sigmoid_3 = model_3.loss_stats()
    plot_learning_curve([train_loss_sigmoid_3[:, 0], test_loss_sigmoid_3[:, 0]],
                        [train_loss_sigmoid_3[:, 1], test_loss_sigmoid_3[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-4",
                        path="../outputs/train_test_loss_sigmoid_1e-4.png")

    print("------------------------------------------------------------------------")
    print("MLP with Sigmoid activation; learning rate=1e-2; 250 iteration scheduling; Decay=0.85")
    print("------------------------------------------------------------------------")
    model_4 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["sigmoid", "sigmoid", "sigmoid", "linear"],
                  learning_rate=1e-2, learning_rate_decay=0.85,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_4.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_4.eval(X_test, y_test)
    train_loss_sigmoid_4, test_loss_sigmoid_4 = model_4.loss_stats()
    plot_learning_curve([train_loss_sigmoid_4[:, 0], test_loss_sigmoid_4[:, 0]],
                        [train_loss_sigmoid_4[:, 1], test_loss_sigmoid_4[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-2; alpha decay=0.85",
                        path="../outputs/train_test_loss_sigmoid_1e-2_sched.png")

    print("------------------------------------------------------------------------")
    print("MLP with Sigmoid activation; learning rate=1e-2; No scheduling; No momentum")
    print("------------------------------------------------------------------------")
    model_5 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["sigmoid", "sigmoid", "sigmoid", "linear"],
                  learning_rate=1e-2, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.)
    model_5.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_5.eval(X_test, y_test)
    train_loss_sigmoid_5, test_loss_sigmoid_5 = model_5.loss_stats()
    plot_learning_curve([train_loss_sigmoid_5[:, 0], test_loss_sigmoid_5[:, 0]],
                        [train_loss_sigmoid_5[:, 1], test_loss_sigmoid_5[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-2; No momentum",
                        path="../outputs/train_test_loss_sigmoid_1e-2_nomomentum.png")

    print("------------------------------------------------------------------------")
    print("MLP with ReLU activation; learning rate=1e-2; No scheduling")
    print("------------------------------------------------------------------------")
    model_6 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["relu", "relu", "relu", "linear"],
                  learning_rate=1e-2, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_6.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_6.eval(X_test, y_test)
    train_loss_relu_6, test_loss_relu_6 = model_6.loss_stats()
    plot_learning_curve([train_loss_relu_6[:, 0], test_loss_relu_6[:, 0]],
                        [train_loss_relu_6[:, 1], test_loss_relu_6[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for ReLU activation; alpha = 1e-2",
                        path="../outputs/train_test_loss_relu_1e-2.png")

    print("------------------------------------------------------------------------")
    print("MLP with ReLU activation; learning rate=1e-3; No scheduling")
    print("------------------------------------------------------------------------")
    model_7 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["relu", "relu", "relu", "linear"],
                  learning_rate=1e-3, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_7.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_7.eval(X_test, y_test)
    train_loss_relu_7, test_loss_relu_7 = model_7.loss_stats()
    plot_learning_curve([train_loss_relu_7[:, 0], test_loss_relu_7[:, 0]],
                        [train_loss_relu_7[:, 1], test_loss_relu_7[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for ReLU activation; alpha = 1e-3",
                        path="../outputs/train_test_loss_relu_1e-3.png")

    print("------------------------------------------------------------------------")
    print("MLP with ReLU activation; learning rate=1e-4; No scheduling")
    print("------------------------------------------------------------------------")
    model_8 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["relu", "relu", "relu", "linear"],
                  learning_rate=1e-4, learning_rate_decay=1.00,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005,momentum=0.9)
    model_8.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_8.eval(X_test, y_test)
    train_loss_relu_8, test_loss_relu_8 = model_8.loss_stats()
    plot_learning_curve([train_loss_relu_8[:, 0], test_loss_relu_8[:, 0]],
                        [train_loss_relu_8[:, 1], test_loss_relu_8[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-4",
                        path="../outputs/train_test_loss_relu_1e-4.png")

    print("------------------------------------------------------------------------")
    print("MLP with ReLU activation; learning rate=1e-2; 250 iteration scheduling; Decay=0.85")
    print("------------------------------------------------------------------------")
    model_9 = MLP([784, 1000, 500, 250, 10],
                  activation_types=["relu", "relu", "relu", "linear"],
                  learning_rate=1e-2, learning_rate_decay=0.85,
                  learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.9)
    model_9.train(X_train, y_train, batch_size=64, iterations=8000,
                  validation=True, validation_data=(X_test, y_test),
                  validation_interval=200)
    model_9.eval(X_test, y_test)
    train_loss_relu_9, test_loss_relu_9 = model_9.loss_stats()
    plot_learning_curve([train_loss_relu_9[:, 0], test_loss_relu_9[:, 0]],
                        [train_loss_relu_9[:, 1], test_loss_relu_9[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-4; alpha decay = 0.85",
                        path="../outputs/train_test_loss_relu_1e-2_sched.png")

    print("------------------------------------------------------------------------")
    print("MLP with ReLU activation; learning rate=1e-2; No scheduling; No momentum")
    print("------------------------------------------------------------------------")
    model_10 = MLP([784, 1000, 500, 250, 10],
                   activation_types=["relu", "relu", "relu", "linear"],
                   learning_rate=1e-2, learning_rate_decay=1.00,
                   learning_rate_decay_interval=250, l2_reg_lambda=0.005, momentum=0.)
    model_10.train(X_train, y_train, batch_size=64, iterations=8000,
                   validation=True, validation_data=(X_test, y_test),
                   validation_interval=200)
    model_10.eval(X_test, y_test)
    train_loss_relu_10, test_loss_relu_10 = model_10.loss_stats()
    plot_learning_curve([train_loss_relu_10[:, 0], test_loss_relu_10[:, 0]],
                        [train_loss_relu_10[:, 1], test_loss_relu_10[:, 1]],
                        colors=['g-', 'b-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Sigmoid activation; alpha = 1e-4; No momentum",
                        path="../outputs/train_test_loss_relu_1e-2_nomomentum.png")

    # Plotting loss comparison between unscheduled and scheduled alpha decays
    # for Sigmoid activation
    plot_learning_curve([train_loss_sigmoid_1[:, 0], train_loss_sigmoid_4[:, 0]],
                        [train_loss_sigmoid_1[:, 1], train_loss_sigmoid_4[:, 1]],
                        colors=['g-', 'b-'], labels=['Without decay', 'With decay'],
                        title="Effect of learning rate scheduling on training loss evolution (sigmoid activation)",
                        path="../outputs/train_loss_sigmoid_decay_comparison.png")
    plot_learning_curve([test_loss_sigmoid_1[:, 0], test_loss_sigmoid_4[:, 0]],
                        [test_loss_sigmoid_1[:, 1], test_loss_sigmoid_4[:, 1]],
                        colors=['g-', 'b-'], labels=['Without decay', 'With decay'],
                        title="Effect of learning rate scheduling on test loss evolution (sigmoid activation)",
                        path="../outputs/test_loss_sigmoid_decay_comparison.png")

    # Plotting loss comparison between momentum and momentumless SGD
    # for Sigmoid activation
    plot_learning_curve([train_loss_sigmoid_1[:, 0], train_loss_sigmoid_5[:, 0]],
                        [train_loss_sigmoid_1[:, 1], train_loss_sigmoid_5[:, 1]],
                        colors=['g-', 'b-'], labels=['With momentum', 'Without momentum'],
                        title="Effect of momentum on training loss evolution (sigmoid activation)",
                        path="../outputs/train_loss_sigmoid_momentum_comparison.png")
    plot_learning_curve([test_loss_sigmoid_1[:, 0], test_loss_sigmoid_5[:, 0]],
                        [test_loss_sigmoid_1[:, 1], test_loss_sigmoid_5[:, 1]],
                        colors=['g-', 'b-'], labels=['With momentum', 'Without momentum'],
                        title="Effect of momentum on test loss evolution (sigmoid activation)",
                        path="../outputs/test_loss_sigmoid_momentum_comparison.png")

    # Plotting loss comparison between different learning rates
    # for Sigmoid activation
    plot_learning_curve([train_loss_sigmoid_1[:, 0], train_loss_sigmoid_2[:, 0], train_loss_sigmoid_3[:, 0]],
                        [train_loss_sigmoid_1[:, 1], train_loss_sigmoid_2[:, 1], train_loss_sigmoid_3[:, 1]],
                        colors=['g-', 'b-', 'r-'], labels=['alpha = 1e-2', 'alpha = 1e-3', 'alpha = 1e-4'],
                        title="Training loss evolution for different learning rates (sigmoid activation)",
                        path="../outputs/train_loss_sigmoid_lr_comparison.png")
    plot_learning_curve([test_loss_sigmoid_1[:, 0], test_loss_sigmoid_2[:, 0], test_loss_sigmoid_3[:, 0]],
                        [test_loss_sigmoid_1[:, 1], test_loss_sigmoid_2[:, 1], test_loss_sigmoid_3[:, 1]],
                        colors=['g-', 'b-', 'r-'], labels=['alpha = 1e-2', 'alpha = 1e-3', 'alpha = 1e-4'],
                        title="Test loss evolution for different learning rates (sigmoid activation)",
                        path="../outputs/test_loss_sigmoid_lr_comparison.png")

    # Plotting loss comparison between unscheduled and scheduled alpha decays
    # for ReLU activation
    plot_learning_curve([train_loss_relu_6[:, 0], train_loss_relu_9[:, 0]],
                        [train_loss_relu_6[:, 1], train_loss_relu_9[:, 1]],
                        colors=['g-', 'b-'], labels=['Without decay', 'With decay'],
                        title="Effect of learning rate scheduling on training loss evolution (ReLU activation)",
                        path="../outputs/train_loss_relu_decay_comparison.png")
    plot_learning_curve([test_loss_relu_6[:, 0], test_loss_relu_9[:, 0]],
                        [test_loss_relu_6[:, 1], test_loss_relu_9[:, 1]],
                        colors=['g-', 'b-'], labels=['Without decay', 'With decay'],
                        title="Effect of learning rate scheduling on test loss evolution (ReLU activation)",
                        path="../outputs/test_loss_relu_decay_comparison.png")

    # Plotting loss comparison between momentum and momentumless SGD
    # for ReLU activation
    plot_learning_curve([train_loss_relu_6[:, 0], train_loss_relu_10[:, 0]],
                        [train_loss_relu_6[:, 1], train_loss_relu_10[:, 1]],
                        colors=['g-', 'b-'], labels=['With momentum', 'Without momentum'],
                        title="Effect of momentum on training loss evolution (ReLU activation)",
                        path="../outputs/train_loss_relu_momentum_comparison.png")
    plot_learning_curve([test_loss_relu_6[:, 0], test_loss_relu_10[:, 0]],
                        [test_loss_relu_6[:, 1], test_loss_relu_10[:, 1]],
                        colors=['g-', 'b-'], labels=['With momentum', 'Without momentum'],
                        title="Effect of momentum on test loss evolution (ReLU activation)",
                        path="../outputs/test_loss_relu_momentum_comparison.png")

    # Plotting loss comparison between different learning rates
    # for ReLU activation
    plot_learning_curve([train_loss_relu_6[:, 0], train_loss_relu_7[:, 0], train_loss_relu_8[:, 0]],
                        [train_loss_relu_6[:, 1], train_loss_relu_7[:, 1], train_loss_relu_8[:, 1]],
                        colors=['g-', 'b-', 'r-'], labels=['alpha = 1e-2', 'alpha = 1e-3', 'alpha = 1e-4'],
                        title="Training loss evolution for different learning rates (ReLU activation)",
                        path="../outputs/train_loss_relu_lr_comparison.png")
    plot_learning_curve([test_loss_relu_6[:, 0], test_loss_relu_7[:, 0], test_loss_relu_8[:, 0]],
                        [test_loss_relu_6[:, 1], test_loss_relu_7[:, 1], test_loss_relu_8[:, 1]],
                        colors=['g-', 'b-', 'r-'], labels=['alpha = 1e-2', 'alpha = 1e-3', 'alpha = 1e-4'],
                        title="Test loss evolution for different learning rates (ReLU activation)",
                        path="../outputs/test_loss_relu_lr_comparison.png")

    # Plotting loss comparison between sigmoid and ReLU activations
    plot_learning_curve([train_loss_sigmoid_1[:, 0], train_loss_relu_6[:, 0]],
                        [train_loss_sigmoid_1[:, 1], train_loss_relu_6[:, 1]],
                        colors=['g-', 'b-'], labels=['Sigmoid', 'ReLU'],
                        title="Training loss comparison between sigmoid and ReLU activations",
                        path="../outputs/train_sigmoid_relu_comparison.png")
    plot_learning_curve([test_loss_sigmoid_1[:, 0], test_loss_relu_6[:, 0]],
                        [test_loss_sigmoid_1[:, 1], test_loss_relu_6[:, 1]],
                        colors=['g-', 'b-'], labels=['Sigmoid', 'ReLU'],
                        title="Test loss comparison between sigmoid and ReLU activations",
                        path="../outputs/test_sigmoid_relu_comparison.png")

    # Randomly choosing 20 images and predicting their top three predictions
    ids = np.random.choice(X_test.shape[0], 20)
    X_samples = X_train[ids]
    pred_samples = model_1.predict(X_samples, num_predictions=3)
    generate_image_outputs(X_samples, pred_samples, path="../outputs/sigmoid")
    pred_samples = model_6.predict(X_samples, num_predictions=3)
    generate_image_outputs(X_samples, pred_samples, path="../outputs/relu")
