import numpy as np

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01): dx = np.ones_like(x); dx[x < 0] = alpha; return dx

def elu(x, alpha=1.0): return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
def elu_derivative(x, alpha=1.0): return np.where(x >= 0, 1, elu(x, alpha) + alpha)

def selu(x): scale, alpha = 1.0507, 1.67326; return scale * elu(x, alpha)
def selu_derivative(x): scale, alpha = 1.0507, 1.67326; return scale * elu_derivative(x, alpha)

def swish(x): return x * sigmoid(x)
def swish_derivative(x): s = sigmoid(x); return s + x * s * (1 - s)

def mish(x): return x * np.tanh(np.log1p(np.exp(x)))
def mish_derivative(x): sp = sigmoid(x); tanh_sp = np.tanh(np.log1p(np.exp(x))); return tanh_sp + x * sp * (1 - tanh_sp**2)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def linear(x): return x
def linear_derivative(x): return np.ones_like(x)

def softmax(x): exps = np.exp(x - np.max(x, axis=-1, keepdims=True)); return exps / np.sum(exps, axis=-1, keepdims=True)
def softmax_derivative(x):
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def gelu_derivative(x): sqrt_2_pi = np.sqrt(2 / np.pi); x3 = x ** 3; tanh_term = np.tanh(sqrt_2_pi * (x + 0.044715 * x3)); sech2 = 1 - tanh_term ** 2; return 0.5 * tanh_term + 0.5 * x * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * x ** 2) + 0.5

def bent_identity(x): return (np.sqrt(x**2 + 1) - 1) / 2 + x
def bent_identity_derivative(x): return x / (2 * np.sqrt(x**2 + 1)) + 1

def gaussian(x): return np.exp(-x**2)
def gaussian_derivative(x): return -2 * x * np.exp(-x**2)

def softplus(x): return np.log1p(np.exp(x))
def softplus_derivative(x): return sigmoid(x)


activation_functions = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (lambda x: leaky_relu(x, 0.01), lambda x: leaky_relu_derivative(x, 0.01)),
    'elu': (lambda x: elu(x, 1.0), lambda x: elu_derivative(x, 1.0)),
    'selu': (selu, selu_derivative),
    'swish': (swish, swish_derivative),
    'mish': (mish, mish_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative),
    'softmax': (softmax, softmax_derivative),
    'gelu': (gelu, gelu_derivative),
    'bent_identity': (bent_identity, bent_identity_derivative),
    'gaussian': (gaussian, gaussian_derivative),
    'softplus': (softplus, softplus_derivative),
}

class Activation:
    def forward(self, x): raise NotImplementedError
    def backward(self, x): raise NotImplementedError


class ReLU(Activation):
    def forward(self, x): return relu(x)
    def backward(self, z, grad_output): return relu_derivative(z) * grad_output

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01): self.alpha = alpha
    def forward(self, x): return leaky_relu(x, self.alpha)
    def backward(self, z, grad_output): return leaky_relu_derivative(z, self.alpha) * grad_output

class ELU(Activation):
    def __init__(self, alpha=1.0): self.alpha = alpha
    def forward(self, x): return elu(x, self.alpha)
    def backward(self, z, grad_output): return elu_derivative(z, self.alpha) * grad_output

class SELU(Activation):
    def forward(self, x): return selu(x)
    def backward(self, z, grad_output): return selu_derivative(z) * grad_output

class Swish(Activation):
    def forward(self, x): return swish(x)
    def backward(self, z, grad_output): return swish_derivative(z) * grad_output

class Mish(Activation):
    def forward(self, x): return mish(x)
    def backward(self, z, grad_output): return mish_derivative(z) * grad_output

class Sigmoid(Activation):
    def forward(self, x): return sigmoid(x)
    def backward(self, z, grad_output): return sigmoid_derivative(z) * grad_output

class Tanh(Activation):
    def forward(self, x): return tanh(x)
    def backward(self, z, grad_output): return tanh_derivative(z) * grad_output

# class Softmax(Activation):
#     def forward(self, x): return softmax(x)
#     def backward(self, z, grad_output): return softmax_derivative(z) * grad_output
class Softmax(Activation):
    def forward(self, x): return softmax(x)
    def backward(self, z, grad_output):
        print(f"Softmax backward - grad_output shape: {grad_output.shape}")
        return grad_output

class GELU(Activation):
    def forward(self, x): return gelu(x)
    def backward(self, z, grad_output): return gelu_derivative(z) * grad_output

class Linear(Activation):
    def forward(self, x): return linear(x)
    def backward(self, z, grad_output): return linear_derivative(z) * grad_output

class Softplus(Activation):
    def forward(self, x): return softplus(x)
    def backward(self, z, grad_output): return softplus_derivative(z) * grad_output


def get_activation(name, alpha=1.0):
    name = name.lower() if name else None
    return {
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, alpha * x),
        'elu': lambda x: elu(x, alpha),
        'selu': lambda x: selu(x, alpha),
        'swish': swish,
        'mish': mish,
        'sigmoid': sigmoid,
        'tanh': np.tanh,
        'linear': lambda x: x,
        'softmax': softmax,
        'gelu': gelu,
        'bent_identity': bent_identity,
        'gaussian': gaussian,
        'softplus': softplus,
        None: lambda x: x
    }[name]


def get_activation_derivative(name, alpha=1.0):
    name = name.lower() if name else None
    return {
        'relu': lambda x: (x > 0).astype(float),
        'leaky_relu': lambda x: np.where(x > 0, 1, alpha),
        'elu': lambda x: np.where(x >= 0, 1, alpha * np.exp(x)),
        'selu': lambda x: 1.0507 * np.where(x >= 0, 1, alpha * np.exp(x)),
        'swish': lambda x: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)),
        'mish': lambda x: (
            np.tanh(np.log1p(np.exp(x))) +
            x * sigmoid(x) * (1 - np.tanh(np.log1p(np.exp(x)))**2)
        ),
        'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
        'tanh': lambda x: 1 - np.tanh(x) ** 2,
        'linear': lambda x: np.ones_like(x),
        'softmax': lambda x: x * (1 - x),  # Approximated, not full Jacobian
        'gelu': lambda x: (
            0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) +
            0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)) ** 2) *
            np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        ),
        'bent_identity': lambda x: x / (2 * np.sqrt(x**2 + 1)) + 1,
        'gaussian': lambda x: -2 * x * np.exp(-x**2),
        'softplus': lambda x: sigmoid(x),
        None: lambda x: np.ones_like(x)
    }[name]

def register_activation(name: str, forward_fn, backward_fn):
    activation_functions[name.lower()] = (forward_fn, backward_fn)


__all__ = [
    'get_activation', 'register_activation', 'activation_functions',
    'ReLU', 'LeakyReLU', 'ELU', 'SELU', 'Swish', 'Mish', 'Sigmoid', 'Tanh',
    'Softmax', 'GELU', 'Linear', 'Softplus',
]
