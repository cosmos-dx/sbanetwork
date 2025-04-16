import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        raise NotImplementedError("Subclasses must implement update")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, params, grads):
        for i in range(len(params)):
            for key in params[i]:
                params[i][key] -= self.learning_rate * grads[i][key]

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [{} for _ in params]
            for i in range(len(params)):
                for key in params[i]:
                    self.v[i][key] = np.zeros_like(params[i][key])

        for i in range(len(params)):
            for key in params[i]:
                self.v[i][key] = self.momentum * self.v[i][key] - self.learning_rate * grads[i][key]
                params[i][key] += self.v[i][key]

class NAG(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [{} for _ in params]
            for i in range(len(params)):
                for key in params[i]:
                    self.v[i][key] = np.zeros_like(params[i][key])

        lookahead_params = [{} for _ in params]
        for i in range(len(params)):
            for key in params[i]:
                lookahead_params[i][key] = params[i][key] + self.momentum * self.v[i][key]

        # Assuming grads is a function that takes lookahead_params and returns gradients
        # This needs to be consistent with how your backward pass and gradient calculation work
        # For now, we'll assume grads is already computed based on the original params
        for i in range(len(params)):
            for key in params[i]:
                self.v[i][key] = self.momentum * self.v[i][key] - self.learning_rate * grads[i][key]
                params[i][key] += self.v[i][key]

class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.eps = epsilon
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = [{} for _ in params]
            for i in range(len(params)):
                for key in params[i]:
                    self.h[i][key] = np.zeros_like(params[i][key])

        for i in range(len(params)):
            for key in params[i]:
                self.h[i][key] += grads[i][key] ** 2
                params[i][key] -= self.learning_rate * grads[i][key] / (np.sqrt(self.h[i][key]) + self.eps)

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = epsilon
        self.s = None

    def update(self, params, grads):
        if self.s is None:
            self.s = [{} for _ in params]
            for i in range(len(params)):
                for key in params[i]:
                    self.s[i][key] = np.zeros_like(params[i][key])

        for i in range(len(params)):
            for key in params[i]:
                self.s[i][key] = self.beta * self.s[i][key] + (1 - self.beta) * grads[i][key] ** 2
                params[i][key] -= self.learning_rate * grads[i][key] / (np.sqrt(self.s[i][key]) + self.eps)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [{} for _ in params]
            self.v = [{} for _ in params]
            for i in range(len(params)):
                for key in params[i]:
                    self.m[i][key] = np.zeros_like(params[i][key])
                    self.v[i][key] = np.zeros_like(params[i][key])

        self.t += 1
        for i in range(len(params)):
            for key in params[i]:
                self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * grads[i][key]
                self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * grads[i][key] ** 2

                m_hat = self.m[i][key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][key] / (1 - self.beta2 ** self.t)

                params[i][key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

# Dictionary of available optimizers
optimizers = {
    'sgd': SGD,
    'sgdmomentum': SGDMomentum,
    'nag': NAG,
    'adagrad': AdaGrad,
    'rmsprop': RMSprop,
    'adam': Adam,
}

def get_optimizer(name, **kwargs):
    """
    Retrieve an optimizer instance by name and initialize it with the given parameters.

    Args:
        name (str): Name of the optimizer (e.g., 'adam', 'sgd').
        **kwargs: Parameters to initialize the optimizer.

    Returns:
        object: An instance of the optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' is not supported. Available options: {list(optimizers.keys())}")
    return optimizers[name](**kwargs)

__all__ = [
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'NAG',
    'AdaGrad',
    'RMSprop',
    'Adam',
    'get_optimizer'
]