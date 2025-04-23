"""
================================================================================
Custom Loss Functions for Neural Networks
================================================================================

Author: Abhishek Gupta
Institution: NSUT Delhi
Email: abhishekgupta0118@gmail.com
Github: cosmos-dx
License: MIT License

Description:
------------
This module implements a variety of loss functions using NumPy. It provides a 
base `Loss` class with several commonly used loss function subclasses for both 
regression and classification tasks.

The design allows:
- Easy extension to custom loss functions
- Consistent API via the `__call__()` method
- Clean integration with neural network training loops

Included Loss Functions:
------------------------
- Mean Squared Error (MSE) suitable for regression problems
- Mean Absolute Error (MAE) robust alternative to MSE
- Binary Cross-Entropy for binary classification
- Categorical Cross-Entropy for multi-class classification (one-hot)
- Huber Loss combines MSE and MAE to be more robust to outliers

Additional Features:
--------------------
- `get_loss(name, **kwargs)` function to dynamically instantiate loss objects
- Clipping is applied internally in cross-entropy losses to avoid numerical instability
- Huber loss uses a delta parameter to adjust the threshold for squared vs. linear loss

Modules Used:
-------------
- `numpy` for vectorized operations and numerical computation

Usage:
------
Use the loss classes directly in training or retrieve them using `get_loss`:

Example:
    loss_fn = get_loss("mse")
    loss_value = loss_fn(y_true, y_pred)

================================================================================
"""



import numpy as np

class Loss:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

class MAE(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

class BinaryCrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class CategoricalCrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class Huber(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def get_loss(name: str, **kwargs):
    name = name.strip().lower()
    if name == 'mse':
        return MSE()
    elif name == 'mae':
        return MAE()
    elif name in ('binary_crossentropy', 'bce'):
        return BinaryCrossEntropy()
    elif name in ('categorical_crossentropy', 'ce'):
        return CategoricalCrossEntropy()
    elif name == 'huber':
        return Huber(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: '{name}'")

__all__ = [
    'Loss', 'MSE', 'MAE', 'BinaryCrossEntropy', 'CategoricalCrossEntropy', 'Huber', 'get_loss'
]