#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Driver program to train a CNN on MNIST dataset.
"""
from math import log10

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input
from keras.models import load_model, Model
from keras.utils import to_categorical
from scipy.ndimage.filters import gaussian_filter

from custom_callbacks import LossHistory
from custom_models import baseline_model, two_conv_layer_model, two_conv_one_dense_layer_model
from utils import preprocess_image_data, get_iter_batch, plot_learning_curve, generate_image_outputs, \
    generate_noisy_outputs

# Initializing essential constants
batch_size = 128
num_classes = 10
epochs = 1
img_rows, img_cols = 28, 28
num_iter = 101

# Initializing essential global variables
input_shape = None
X_train, y_train_labels, y_train, X_test, y_test_labels, y_test = None, None, None, None, None, None


def normalize_tensor(x):
    """ Utility function to normalize a tensor by its L2 norm

        Params:
            x: Tensorflow tensor

        Returns:
            tensor: Normalized tensor

    """
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_data():
    """ Helper function to load and initialize data
    """
    global input_shape, X_train, y_train_labels, y_train, X_test, y_test_labels, y_test
    (X_train, y_train_labels), (X_test, y_test_labels) = mnist.load_data()
    X_train, X_test, input_shape = preprocess_image_data(X_train, X_test, img_rows, img_cols, K)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train_labels, num_classes)
    y_test = to_categorical(y_test_labels, num_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


def question_1():
    global input_shape, X_train, y_train_labels, y_train, X_test, y_test_labels, y_test

    print("------------------------------------------------------------------------")
    print("Baseline Model")
    print("------------------------------------------------------------------------")
    model1 = baseline_model(input_shape, num_classes)
    loss_callback_1 = LossHistory((X_test, y_test))
    model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test),
               callbacks=[loss_callback_1])
    model1.save('model1.h5')
    plot_learning_curve([loss_callback_1.train_indices, loss_callback_1.test_indices],
                        [loss_callback_1.train_losses, loss_callback_1.test_losses],
                        colors=['g-', 'm-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for Baseline Model",
                        path="../outputs/q1/plots/train_test_loss_baseline.png",
                        axlabels=["Iterations", "Loss"])
    plot_learning_curve([loss_callback_1.test_indices],
                        [loss_callback_1.test_acc],
                        colors=['c-'], labels=['Test Accuracy'],
                        title="Accuracy evolution for Baseline Model",
                        path="../outputs/q1/plots/test_acc_baseline.png",
                        axlabels=["Iterations", "Accuracy"])

    print("------------------------------------------------------------------------")
    print("2 conv layer model")
    print("------------------------------------------------------------------------")
    model2 = two_conv_layer_model(input_shape, num_classes)
    loss_callback_2 = LossHistory((X_test, y_test))
    model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test),
               callbacks=[loss_callback_2])
    model2.save('model2.h5')
    plot_learning_curve([loss_callback_2.train_indices, loss_callback_2.test_indices],
                        [loss_callback_2.train_losses, loss_callback_2.test_losses],
                        colors=['g-', 'm-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for 2 conv layered Model",
                        path="../outputs/q1/plots/train_test_loss_2_conv.png",
                        axlabels=["Iterations", "Loss"])
    plot_learning_curve([loss_callback_1.test_indices],
                        [loss_callback_1.test_acc],
                        colors=['c-'], labels=['Test Accuracy'],
                        title="Accuracy evolution for 2 conv layered Model",
                        path="../outputs/q1/plots/test_acc_2_conv.png",
                        axlabels=["Iterations", "Accuracy"])

    print("------------------------------------------------------------------------")
    print("2 conv layer + 1 hidden dense layer model")
    print("------------------------------------------------------------------------")
    model3 = two_conv_one_dense_layer_model(input_shape, num_classes)
    loss_callback_3 = LossHistory((X_test, y_test))
    model3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test),
               callbacks=[loss_callback_3])
    model3.save('model3.h5')
    plot_learning_curve([loss_callback_3.train_indices, loss_callback_3.test_indices],
                        [loss_callback_3.train_losses, loss_callback_3.test_losses],
                        colors=['g-', 'm-'], labels=['Train loss', 'Test loss'],
                        title="Loss evolution for 2 Conv + 1 Dense layer config",
                        path="../outputs/q1/plots/train_test_loss_2_conv_1_dense.png",
                        axlabels=["Iterations", "Loss"])
    plot_learning_curve([loss_callback_3.test_indices],
                        [loss_callback_3.test_acc],
                        colors=['c-'], labels=['Test Accuracy'],
                        title="Accuracy evolution for 2 conv + 1 dense config",
                        path="../outputs/q1/plots/test_acc_2_conv_1_dense.png",
                        axlabels=["Iterations", "Accuracy"])

    ids = np.random.choice(X_test.shape[0], 20)
    X_samples = X_train[ids]
    pred_samples_1 = model1.predict(X_samples)
    generate_image_outputs(X_samples, np.argmax(pred_samples_1, axis=1), path="../outputs/q1/predictions/baseline")
    pred_samples_2 = model2.predict(X_samples)
    generate_image_outputs(X_samples, np.argmax(pred_samples_2, axis=1), path="../outputs/q1/predictions/2_conv")
    pred_samples_3 = model3.predict(X_samples)
    generate_image_outputs(X_samples, np.argmax(pred_samples_3, axis=1),
                           path="../outputs/q1/predictions/2_conv_1_dense")


def question_2():
    global input_shape, X_train, y_train_labels, y_train, X_test, y_test_labels, y_test
    model3 = load_model('model3.h5')
    model3.trainable = False
    learning_rate = 0.01
    validation_interval = 10

    # Iterating over each of the 10 classes for generating adversarial examples
    for _label in range(0, num_classes):
        print("------------------------------------------------------------------------")
        print("Adversarial examples for label " + str(_label))
        print("------------------------------------------------------------------------")

        # y_eval is a dummy matrix useful for evaluating categorical crossentropy loss
        y_eval = to_categorical(np.full((batch_size, 1), _label, dtype=int), num_classes=num_classes)
        # y_fool is the duplicate label meant to fool the network and generate adversarial examples
        y_fool = to_categorical(np.full((y_train_labels.shape[0], 1), _label, dtype=int), num_classes=num_classes)

        batch = get_iter_batch(X_test, y_fool, batch_size, num_iter)

        # initializing a 28 x 28 matrix for noise
        noise = np.zeros((1, 28, 28, 1))

        # new functional model to add noise and predict output using existing trained model
        input1 = Input(shape=(img_rows, img_cols, 1))
        input2 = Input(shape=(img_rows, img_cols, 1))
        sum_inp = keras.layers.add([input1, input2])
        op = model3(sum_inp)
        noise_model = Model(inputs=[input1, input2], outputs=op)

        # calculating gradient
        a_loss = K.categorical_crossentropy(noise_model.output, y_eval)
        grad = K.gradients(a_loss, noise_model.input[1])[0]
        grad = K.mean(normalize_tensor(grad), axis=0)

        # custom keras backend function that takes in two inputs and yields noise output,
        # loss and gradient
        custom_iterate = K.function([input1, input2], [noise_model.output, a_loss, grad])

        train_indices, train_loss, test_indices, test_loss, test_acc = [], [], [], [], []
        ctr = 0

        # Batch wise manual gradient descent for learning adversarial noise
        for _batch in batch:
            X_actual, y_actual = _batch
            output, loss, grads = custom_iterate([X_actual, noise])

            # Validating at specific intervals
            if (ctr % validation_interval == 0):
                noise_test = np.zeros(X_test.shape) + noise[0]
                preds_test = noise_model.predict([X_test, noise_test])
                _test_acc = float(np.where(np.argmax(preds_test, axis=1) == _label)[0].shape[0]) / float(
                    preds_test.shape[0])
                _test_loss = np.mean(loss)
                test_indices.append(ctr)
                test_loss.append(_test_loss)
                test_acc.append(_test_acc)

            train_indices.append(ctr)
            train_loss.append(np.mean(loss))

            # Gradient update
            noise = noise - learning_rate * np.array(grads)

            line = ("Iteration " + str(ctr + 1).rjust(int(log10(num_iter) + 1))
                    + "/" + str(num_iter)
                    + " complete.       Train Loss: %0.10f       " % np.mean(loss))
            print(line)
            ctr = ctr + 1

        noise_test = np.zeros(X_test.shape) + noise[0]
        preds = noise_model.predict([X_test, noise_test])
        print(
        "Accuracy: " + str(float(np.where(np.argmax(preds, axis=1) == _label)[0].shape[0]) / float(preds.shape[0])))

        # Visualizing each of the generated noises
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(noise.reshape(28, 28), interpolation='nearest', cmap="gray")
        plt.savefig("../outputs/q2/visualizations/sample_" + str(_label) + ".png")
        plt.close()

        # Plotting loss and accuracy evolution
        plot_learning_curve([train_indices, test_indices],
                            [train_loss, test_loss],
                            colors=['c-', 'm-'], labels=['Train loss', 'Test loss'],
                            title="Loss evolution for adversarial noise training",
                            path="../outputs/q2/plots/train_test_loss_adversarial_noise_" + str(_label) + ".png",
                            axlabels=["Iterations", "Loss"])
        plot_learning_curve([test_indices],
                            [test_acc],
                            colors=['r-'], labels=['Test Accuracy'],
                            title="Accuracy evolution for adversarial noise training",
                            path="../outputs/q2/plots/test_acc_adversarial_noise_" + str(_label) + ".png",
                            axlabels=["Iterations", "Accuracy"])

        # Predicting for a random set of 9 adversarial images
        ids = np.random.choice(X_test.shape[0], 9)
        X_samples = X_test[ids]
        noise_sample = np.zeros(X_samples.shape) + noise[0]
        pred_samples = noise_model.predict([X_samples, noise_sample])
        actual_samples = model3.predict(X_samples)
        generate_noisy_outputs(X_samples + noise_sample, np.argmax(actual_samples, axis=1),
                               np.argmax(pred_samples, axis=1), path="../outputs/q2/predictions/" + str(_label))


def question_3():
    global input_shape, X_train, y_train_labels, y_train, X_test, y_test_labels, y_test
    model = load_model('model3.h5')
    model.trainable = False

    # Custom model that inputs 28 x 28 matrices and outputs logits (without softmax)
    visualize_model = Model(inputs=model.input, outputs=model.get_layer("logits").output)

    for _label in range(0, num_classes):
        print("------------------------------------------------------------------------")
        print("Synthetic image visualization for label " + str(_label))
        print("------------------------------------------------------------------------")
        y_temp = [_label]
        y_temp = to_categorical(y_temp, num_classes)

        # Setting cost to be the respective output neurons
        cost = visualize_model.output[:, _label]
        # Gradient calculation for the cost
        grad = K.mean(K.gradients(cost, visualize_model.input)[0], axis=0)

        # Custom keras backend function that inputs the images and returns the cost and gradient
        custom_iterate = K.function([model.input], [visualize_model.output[:, _label], grad])

        # Initializing a gaussian distribution centred around 128
        X_init = np.random.normal(loc=128., scale=50., size=(1, 28, 28, 1))
        X_init /= 255.

        costs = []
        iter_indices = []

        # Batch wise gradient ascent for learning X_init
        for i in range(num_iter):
            cost, grads = custom_iterate([X_init])
            sigma = (i + 1) * 4 / (num_iter + 0.5)
            step_size = 1.0 / np.std(grads)
            costs.append(cost[0])
            iter_indices.append(i)

            # Smoothening using a Gaussian filter
            grads = gaussian_filter(grads, sigma)
            # Gradient update
            X_init = (1 - 0.0001) * X_init + step_size * np.array(grads)

            line = ("Iteration " + str(i + 1).rjust(int(log10(num_iter) + 1))
                    + "/" + str(num_iter)
                    + " complete.       Cost: %0.10f       " % cost[0])
            print(line)

        # Visualizing the input image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(X_init.reshape(28, 28), interpolation='nearest', cmap="gray")
        plt.savefig("../outputs/q3/visualizations/max_output_" + str(_label) + ".png")
        plt.close()

        plot_learning_curve([iter_indices],
                            [costs],
                            colors=['b-'], labels=['Cost'],
                            title="Cost evolution over optimization iterations",
                            path="../outputs/q3/plots/cost_output_" + str(_label) + ".png",
                            axlabels=["Iterations", "Cost"])

    # Custom model that inputs 28 x 28 image matrices and outputs 2nd maxpooling layer
    visualize_model = Model(inputs=model.input, outputs=model.get_layer("maxpooling2").output)
    for _id in range(15):
        print("------------------------------------------------------------------------")
        print("Synthetic image visualization for central neuron of filter " + str(_id))
        print("------------------------------------------------------------------------")

        # Setting cost as the central neuron of maxpooling layer
        # Since row size and column size (7, 7) is odd, we do row/2 and column/2
        cost = visualize_model.output[:, visualize_model.output.get_shape()[1] / 2,
               visualize_model.output.get_shape()[2] / 2, _id]
        grad = K.mean(K.gradients(cost, visualize_model.input)[0], axis=0)
        custom_iterate = K.function([model.input], [cost, grad])
        X_init = np.random.normal(loc=128., scale=50., size=(1, 28, 28, 1))
        X_init /= 255.

        # Batch wise gradient ascent for learning X_init
        for i in range(num_iter):
            cost, grads = custom_iterate([X_init])
            sigma = (i + 1) * 4 / (num_iter + 0.5)
            step_size = 1.0 / np.std(grads)
            grads = gaussian_filter(grads, sigma)

            # Gradient update
            X_init = (1 - 0.0001) * X_init + step_size * np.array(grads)

            line = ("Iteration " + str(i + 1).rjust(int(log10(num_iter) + 1))
                    + "/" + str(num_iter)
                    + " complete.       Cost: %0.10f       " % cost[0])
            print(line)

        # Plotting X_init for each of the filter optimizations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(X_init.reshape(28, 28), interpolation='nearest', cmap="gray")
        plt.text(0.5, 0.05, 'Filter: ' + str(_id), fontsize=28,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, color='white')
        plt.savefig("../outputs/q3/visualizations/max_filter_" + str(_id) + ".png")
        plt.close()


if __name__ == "__main__":
    load_data()
    question_1()
    question_2()
    question_3()
