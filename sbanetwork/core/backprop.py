"""
backprop.py
-----------

This module defines derivatives of common loss functions and metrics used during the backward pass of training in the SBA Network (Sparse Biological-inspired Adaptive Network) library.

Loss function derivatives are essential for computing gradients during backpropagation. This module supports both categorical and binary classification tasks, as well as regression loss derivatives. It also includes commonly used evaluation metrics.

Functions:
    - categorical_crossentropy_derivative(y_true, y_pred)
    - binary_crossentropy_derivative(y_true, y_pred)
    - mse_derivative(y_true, y_pred)
    - mae_derivative(y_true, y_pred)
    - accuracy_score(y_true, y_pred)
    - binary_accuracy(y_true, y_pred)
    - get_loss_derivative(name)
    - get_metric(name)

Notes
-----
Author: Abhishek Gupta  
Library: sbanetwork  
Email: abhishekgupta0118@gmail.com
Github: cosmos-dx


The `sbanetwork` library provides a biologically inspired, modular approach to building and training sparse neural networks. This `backprop.py` module enables flexible loss and metric handling, crucial for gradient-based optimization. It supports dynamic loss retrieval and evaluation during training for improved model adaptability and performance monitoring.
"""

import numpy as np

def categorical_crossentropy_derivative(y_true, y_pred, epsilon=1e-8):
    """
    Derivative of categorical cross-entropy with respect to the SOFTMAX OUTPUT.
    """
    return y_pred - y_true

def binary_crossentropy_derivative(y_true, y_pred, epsilon=1e-12):
    """
    Derivative of binary cross-entropy with respect to the output.
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size

def mse_derivative(y_true, y_pred):
    """
    Derivative of Mean Squared Error with respect to the output.
    """
    return 2 * (y_pred - y_true) / y_true.size

def mae_derivative(y_true, y_pred):
    """
    Derivative of Mean Absolute Error with respect to the output.
    """
    return np.sign(y_pred - y_true) / y_true.size

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score.
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculates binary accuracy.
    """
    return np.mean(((y_pred > threshold).astype(int)) == y_true)

def get_loss_derivative(name: str):
    name = name.strip().lower()
    if name in ('categorical_crossentropy', 'ce'):
        return categorical_crossentropy_derivative
    elif name in ('binary_crossentropy', 'bce'):
        return binary_crossentropy_derivative
    elif name == 'mse':
        return mse_derivative
    elif name == 'mae':
        return mae_derivative
    else:
        raise ValueError(f"Unsupported loss derivative: '{name}'")

def get_metric(name: str):
    name = name.strip().lower()
    if name == 'accuracy':
        return accuracy_score
    elif name == 'binary_accuracy':
        return binary_accuracy
    else:
        raise ValueError(f"Unsupported metric: '{name}'")

__all__ = [
    'categorical_crossentropy_derivative', 'binary_crossentropy_derivative',
    'mse_derivative', 'mae_derivative', 'accuracy_score', 'binary_accuracy',
    'get_loss_derivative', 'get_metric'
]