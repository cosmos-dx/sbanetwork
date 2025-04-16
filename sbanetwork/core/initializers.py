import numpy as np

class Initializer:
    def initialize(self, shape):
        raise NotImplementedError

    def __call__(self, shape):
        return self.initialize(shape)


class Zeros(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    def initialize(self, shape):
        return np.ones(shape)


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape):
        return np.random.normal(self.mean, self.stddev, size=shape)


class XavierUniform(Initializer):
    def initialize(self, shape):
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)


class XavierNormal(Initializer):
    def initialize(self, shape):
        fan_in, fan_out = shape[0], shape[1]
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)


class HeUniform(Initializer):
    def initialize(self, shape):
        fan_in = shape[0]
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)


class HeNormal(Initializer):
    def initialize(self, shape):
        fan_in = shape[0]
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=shape)


class LeCunNormal(Initializer):
    def initialize(self, shape):
        fan_in = shape[0]
        stddev = np.sqrt(1 / fan_in)
        return np.random.normal(0, stddev, size=shape)


class LeCunUniform(Initializer):
    def initialize(self, shape):
        fan_in = shape[0]
        limit = np.sqrt(3 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)


class Orthogonal(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def initialize(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q


class LSUV(Initializer):
    def initialize(self, shape):
        weights = np.random.normal(0, 1, shape)
        for _ in range(10):
            norm = np.std(weights)
            if norm < 1e-3:
                break
            weights /= norm
        return weights


initializers = {
    'zeros': Zeros,
    'ones': Ones,
    'random_normal': RandomNormal,
    'xavier_uniform': XavierUniform,
    'xavier_normal': XavierNormal,
    'he_uniform': HeUniform,
    'he_normal': HeNormal,
    'lecun_normal': LeCunNormal,
    'lecun_uniform': LeCunUniform,
    'orthogonal': Orthogonal,
    'lsuv': LSUV,
    'glorot_uniform': XavierUniform, 
    'glorot_normal': XavierNormal,   
}
def get_initializer(name, **kwargs):
    """
    Returns a callable initializer instance.

    Args:
        name (str): Name of the initializer.
        **kwargs: Parameters for the initializer (e.g., gain for Orthogonal).

    Returns:
        Initializer instance (callable)

    Raises:
        ValueError: If the initializer name is invalid.
    """
    name = name.lower()
    if name not in initializers:
        raise ValueError(f"Initializer '{name}' is not supported. Available options are: {list(initializers.keys())}")
    return initializers[name](**kwargs)
