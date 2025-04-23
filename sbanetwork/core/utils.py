# ================================================================
# utils.py
# ================================================================
# Author: Abhishek Gupta
# Description:
# This script defines various data scaling and normalization techniques 
# commonly used in machine learning to preprocess and standardize datasets.
# The classes included are:
# - StandardScaler: Standardizes the data by subtracting the mean and dividing by 
#   the standard deviation (z-score normalization).
# - MinMaxScaler: Scales the data to a fixed range, typically [0, 1], based on 
#   the minimum and maximum values.
# - Normalizer: Applies either z-score normalization (StandardScaler) or MinMax 
#   scaling (MinMaxScaler) based on the selected method.
#
# In addition, the `clip_gradients` function is provided to apply gradient clipping, 
# which is useful during backpropagation in neural networks to prevent exploding 
# gradients by capping them at a certain threshold.
#
# Usage:
# - You can use the scalers (`StandardScaler`, `MinMaxScaler`, and `Normalizer`) 
#   to fit and transform your dataset as follows:
#     - `fit(X)`: Computes the necessary statistics (mean, std, min, max).
#     - `transform(X)`: Scales the dataset based on the computed statistics.
#     - `fit_transform(X)`: Fits and then transforms the dataset.
#     - `inverse_transform(X)`: Reverses the transformation.
# 
# - The `clip_gradients` function can be used to limit the gradient values, 
#   especially useful during training of neural networks to avoid large gradients.
#
# Libraries used:
# - numpy: For numerical operations, including calculating means, standard deviations,
#   min/max values, and norms.
#
# ================================================================

import numpy as np

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * (self.std + 1e-8) + self.mean

class MinMaxScaler:
    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * (self.max - self.min + 1e-8) + self.min

class Normalizer:
    def __init__(self, method="z-score"):
        self.method = method
        self.scaler = StandardScaler() if method == "z-score" else MinMaxScaler()

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

def clip_gradients(grads, threshold):
    norm = np.linalg.norm(grads)
    if norm > threshold:
        return grads * threshold / norm
    return grads