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