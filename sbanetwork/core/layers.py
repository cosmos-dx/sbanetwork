"""
================================================================================
Custom Neural Network Layers Library
================================================================================

Author: Abhishek Gupta
Institution: NSUT Delhi
Email: abhishekgupta0118@gmail.com
Github: cosmos-dx
License: MIT License

Description:
------------
This module is a foundational library for building neural network layers 
from scratch using NumPy. It includes a base `Layer` class and specific 
implementations like `Dense` (fully connected layer) and `Dropout`.

The design allows:
- Clean, modular layer definitions
- Manual weight initialization and regularization
- Full control over the forward and backward pass
- Integration with custom training loops
- Educational insight into deep learning internals

Supported Features:
-------------------
- Fully Connected Dense Layer with custom activation functions
- Dropout regularization for training
- Pluggable initializers and regularizers
- Backpropagation and gradient handling

This library is ideal for:
- Educational projects and research
- Lightweight experiments without major dependencies (e.g., TensorFlow/PyTorch)
- Extending to convolutional or recurrent layers

Modules Used:
-------------
- `numpy` for numerical computation
- Custom modules:
    - `initializers` to initialize weights
    - `activations`  for activation functions and derivatives
    - `regularizers`for L1, L2, or custom regularization

Usage:
------
This module should be imported as part of a neural network framework 
and used to compose a model by stacking layers. You can build, train, 
and evaluate models using these core building blocks.

Example:
    dense1 = Dense(units=64, activation='relu')
    dropout = Dropout(rate=0.5)
    dense2 = Dense(units=10, activation='softmax')

================================================================================
"""


import numpy as np
from . import initializers
from . import activations
from . import regularizers


class Layer:
    def __init__(self, name=None, trainable=True):
        self.name = name
        self.trainable = trainable
        self.params = {}  
        self.grads = {}   
        self.regularizers = {} 
        self.input_shape = None
        self.output_shape = None
        self.input = None
        self.linear_output = None 

    def build(self, input_shape):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        raise NotImplementedError

    def add_weight(self, name, shape, initializer='random_normal', regularizer=None):
        if name not in self.params:
            self.params[name] = initializers.get_initializer(initializer)(shape)
            self.grads[name] = np.zeros_like(self.params[name])
            if regularizer:
                self.regularizers[name] = regularizers.get_regularizer(regularizer)

    def get_weights(self):
        return [self.params[name] for name in self.params]

    def get_gradients(self, grad_output):
        raise NotImplementedError("Subclasses must implement get_gradients")

    def apply_regularization(self, name):
        if name in self.regularizers:
            return self.regularizers[name](self.params[name])
        return 0

class Dense(Layer):
    def __init__(self, units, activation=None, kernel_initializer='he_normal',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 name=None, trainable=True):
        super().__init__(name, trainable)
        self.units = units
        self._activation_name = activation.lower() if isinstance(activation, str) else None
        self.activation = activations.get_activation(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = regularizers.get_regularizer(kernel_regularizer)
        self.bias_regularizer = regularizers.get_regularizer(bias_regularizer)
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[-1]
        self.add_weight(name='kernel', shape=(input_dim, self.units), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer)
        self.add_weight(name='bias', shape=(self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer)
        self.output_shape = (input_shape[0], self.units)

    def forward(self, x):
        self.input = x
        self.linear_output = np.dot(x, self.params['kernel']) + self.params['bias']
        return self.activation(self.linear_output)

    def backward(self, grad_output):
        # Special handling for Softmax output layer with CCE loss
        # Assumes grad_output coming in is already dL/dz = y_pred - y_true
        if self._activation_name == 'softmax':
             # We assume the incoming grad_output is dL/dz (logits gradient)
             # directly from the CCE loss derivative calculation (y_pred - y_true)
             grad_linear_output = grad_output
        else:
             # For other activations, apply the chain rule normally: dL/dz = dL/da * da/dz
             # Get the activation derivative function associated with this layer
             # Need a reliable way to get the derivative; using the stored name
             activation_derivative_fn = activations.get_activation_derivative(self._activation_name)
             d_activation = activation_derivative_fn(self.linear_output)
             grad_linear_output = grad_output * d_activation

        # Gradient of the kernel: d(L)/d(kernel) = input.T @ grad_linear_output
        grad_kernel = np.dot(self.input.T, grad_linear_output)
        # Gradient of the bias: d(L)/d(bias) = sum(grad_linear_output, axis=0)
        grad_bias = np.sum(grad_linear_output, axis=0)
        # Gradient with respect to the input of this layer: d(L)/d(input) = grad_linear_output @ kernel.T
        grad_input = np.dot(grad_linear_output, self.params['kernel'].T)

        # Apply regularization gradients (if regularizers exist)
        if self.kernel_regularizer:
             grad_kernel += self.kernel_regularizer.gradient(self.params['kernel'])
        if self.bias_regularizer:
             grad_bias += self.bias_regularizer.gradient(self.params['bias'])

        # Store gradients if layer is trainable
        if self.trainable:
            self.grads['kernel'] = grad_kernel
            self.grads['bias'] = grad_bias

        return grad_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_gradients(self, grad_output=None): # grad_output not needed here as gradients are computed in backward
        return {'kernel': self.grads['kernel'], 'bias': self.grads['bias']}

class Dropout(Layer):
    def __init__(self, rate, name=None, trainable=False): 
        super().__init__(name, trainable)
        self.rate = rate
        self.mask = None
        self.training = True

    def train(self, training):
        self.training = training

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self.mask
        return x

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask
        return grad_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_gradients(self, grad_output=None):
        return {}

# We will define other layer types (Conv2D, MaxPooling, etc.) similarly,
# implementing their specific build, forward, and backward methods.