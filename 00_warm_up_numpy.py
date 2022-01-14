# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

""" Hyper-parameters
"""
if 1 == 1:
    N = 1000  # batch size
    H = 10  # number of neurones in hidden layer
    D_in = 10  # number of imput features
    D_out = 1  # number of output targets
    learning_rate = 1e-6  # learning rate
    epochs = 1000  # number of iterations

""" Create random input and output data
"""
if 1 == 1:
    np.random.seed(0)
    x = np.random.lognormal(0, 0.4, size=(N, D_in))
    y = np.random.beta(1, 3, size=(N, D_out))

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x)
    ax1.set_title('Input features')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(y, label='target', color='red')
    plt.legend()
    plt.show()

""" Randomly initialize weights (one hidden layer)
"""
if 1 == 1:
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

""" Back-propagation procedure
"""

if 1 == 1:
    for t in range(epochs):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        print('step:{:2.2f} - Loss:{}'.format(t, loss))

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

""" Plot results
"""
if 1 ==1:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(0, epochs), y_pred, color='red', label='Prediction')
    ax.plot(np.arange(0, epochs), y, color='blue', label='Data')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(0, epochs), 100*(y_pred -y) /y, color='black', label='error')
    plt.legend()
    plt.show()