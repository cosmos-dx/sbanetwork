import numpy as np

# Each regularization class implements a method to compute the regularization term and its gradient.
# The regularization classes are designed to be used with regression and classification tasks.

class L1:
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def __call__(self, weights):
        """
        Computes the L1 regularization term.
        L1 = lambda * sum(|w|)
        """
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights):
        """
        Computes the gradient of the L1 regularization term with respect to the weights.
        dL1/dw = lambda * sign(w)
        """
        return self.lambda_ * np.sign(weights)


class L2:
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def __call__(self, weights):
        """
        Computes the L2 regularization term (weight decay).
        L2 = 0.5 * lambda * sum(w^2)
        """
        return 0.5 * self.lambda_ * np.sum(weights**2)

    def gradient(self, weights):
        """
        Computes the gradient of the L2 regularization term with respect to the weights.
        dL2/dw = lambda * w
        """
        return self.lambda_ * weights


class MaxNorm:
    def __init__(self, max_value=3):
        self.max_value = max_value

    def __call__(self, weights):
        """
        Computes a penalty if the L2 norm of the weights exceeds max_value.
        Penalty = 0 if ||w|| <= max_value else ||w|| - max_value
        """
        norm = np.linalg.norm(weights)
        return 0 if norm <= self.max_value else norm - self.max_value

    def gradient(self, weights):
        """
        Computes the gradient of the MaxNorm penalty with respect to the weights.
        d(Penalty)/dw = 0 if ||w|| <= max_value else w / ||w||
        """
        norm = np.linalg.norm(weights)
        if norm <= self.max_value:
            return np.zeros_like(weights)
        return weights / norm

    def apply(self, weights):
        """
        Applies the MaxNorm constraint by rescaling weights if their L2 norm exceeds max_value.
        w_new = w * (max_value / ||w||) if ||w|| > max_value else w
        """
        norm = np.linalg.norm(weights)
        if norm > self.max_value:
            weights = weights * (self.max_value / norm)
        return weights


regs = {
    'l1': L1,
    'l2': L2,
    'maxnorm': MaxNorm,
}

def get_regularizer(name, **kwargs):
    """
    Retrieve a Regularizer instance by name and initialize it with the given parameters.

    Args:
        name (str): Name of the regularizer (e.g., 'l1', 'l2', 'maxnorm').
        **kwargs: Parameters to initialize the regularizer.

    Returns:
        object: An instance of the regularizer, or None if name is None.

    Raises:
        ValueError: If the regularizer name is not recognized.
    """
    if name is None:
        return None
    name = name.lower()
    if name not in regs:
        raise ValueError(f"Regularizer '{name}' is not supported. Available options are: {list(regs.keys())}")
    return regs[name](**kwargs)

__all__ = [
    'L1',
    'L2',
    'MaxNorm',
    'get_regularizer'
]