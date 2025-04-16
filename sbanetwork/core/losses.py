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