# ================================================================
# Model.py
# ================================================================
# Author: Abhishek Gupta
# Email: abhishekgupta0118@gmail.com
# Github: cosmos-dx
# Description:
# This script defines a custom deep learning Model class that implements 
# the core functionality of a neural network model, including layer 
# handling, forward and backward passes, loss computation, gradient 
# updates, training, evaluation, and prediction. It provides methods 
# for building, compiling, training, and evaluating models, as well as 
# for exporting model configurations for later use.
#
# Key Features:
# - Custom Layer Handling: Includes the ability to add layers such as 
#   Dense, Dropout, and others to the model.
# - Loss and Metrics: Allows for loss and metrics functions to be 
#   specified and calculated during training and evaluation.
# - Training and Gradient Descent: Implements forward and backward 
#   passes, computes gradients, and applies them using an optimizer.
# - Model Export: Provides functionality to export the model's layer 
#   configuration for persistence.
#
# Libraries used:
# - numpy: For numerical operations (e.g., matrix multiplication, gradient 
#   computations).
# - json: For exporting the model configuration to a JSON file.
# - warnings: For issuing warnings when NaN values are detected in 
#   predictions or gradients.
# - optimizers: Custom implementation for various optimizers like SGD.
# - losses: Custom loss functions used for training the model.
# - backprop: Custom backpropagation logic to calculate gradients.
# - callbacks: Custom callbacks for handling events like early stopping, 
#   learning rate adjustment, etc.
# - layers: Custom layers (e.g., Dense, Dropout) implemented for building 
#   neural networks.

# ================================================================


import numpy as np
import json
import warnings
from . import optimizers
from . import losses
from . import backprop
from . import callbacks
from .layers import Dropout

class Model:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.built = False
        self.optimizer = None
        self.loss_fn = None
        self.loss_derivative_fn = None
        self.metrics = []
        self.history = {}

    def add(self, layer):
        self.layers.append(layer)

    def build(self, input_shape):
        current_shape = input_shape
        for layer in self.layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)
        self.built = True

    def compile(self, optimizer='sgd', loss='mse', learning_rate=0.01, metrics=None):
        if isinstance(optimizer, str):
            self.optimizer = optimizers.get_optimizer(optimizer, learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        if isinstance(loss, str):
            self.loss_fn = losses.get_loss(loss)
            self.loss_derivative_fn = backprop.get_loss_derivative(loss)
        else:
            self.loss_fn = loss
            # Assuming if a function is passed, its derivative needs to be handled externally
            self.loss_derivative_fn = None
        self.metrics = [backprop.get_metric(metric) if isinstance(metric, str) else metric for metric in (metrics if metrics else [])]

    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train(training)
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
             if not hasattr(layer, 'backward'):
                 continue
             grad_output = layer.backward(grad_output)
        return grad_output

    def train_step(self, x_batch, y_batch):
        predictions = self.forward(x_batch, training=True)

        # NaN check for predictions
        if np.isnan(predictions).any():
            warnings.warn("NaN detected in predictions during training step. Stopping.")
            # You might want to return a specific value or raise an error
            return np.nan, {m.__name__: np.nan for m in self.metrics}

        # Calculate loss
        loss = self.loss_fn(y_batch, predictions)

        # NaN check for loss
        if np.isnan(loss):
            warnings.warn("NaN detected in loss during training step. Stopping.")
            return np.nan, {m.__name__: np.nan for m in self.metrics}


        # Calculate the gradient of the loss with respect to the model's output (or logits for CCE/Softmax)
        if self.loss_derivative_fn is None:
             raise RuntimeError("Loss derivative function is not set. Compile the model with a valid loss string or provide derivative.")

        # grad_output here is dL/da (or dL/dz if CCE/Softmax coupled derivative is used)
        grad_output = self.loss_derivative_fn(y_batch, predictions)

        # NaN check for initial gradient
        if np.isnan(grad_output).any():
            warnings.warn("NaN detected in initial gradient during training step. Stopping.")
            return loss, {m.__name__: np.nan for m in self.metrics}


        # Backward pass - Calculates gradients and stores them in layer.grads
        _ = self.backward(grad_output) 
        # Get gradients for all trainable layers and apply clipping
        trainable_layers = [layer for layer in self.layers if layer.trainable]
        layer_params = []
        layer_grads = []
        total_grad_norm = 0.0
        max_norm_clip = 1.0 # Define your clipping threshold

        for layer in trainable_layers:
            if not layer.params or not layer.grads: 
                continue

            layer_params.append(layer.params)
            clipped_grads_dict = {}
            for key in layer.grads:
                grad = layer.grads[key]
                if np.isnan(grad).any():
                     warnings.warn(f"NaN detected in gradients for layer '{layer.name}', param '{key}'. Stopping.")
                     return loss, {m.__name__: np.nan for m in self.metrics}

                # --- Gradient Clipping (per parameter array) ---
                norm = np.linalg.norm(grad)
                total_grad_norm += norm**2
                if norm > max_norm_clip:
                    clipped_grad = grad * (max_norm_clip / (norm + 1e-7)) # Add epsilon for stability
                    clipped_grads_dict[key] = clipped_grad
                    # print(f"Clipping grad for {layer.name}/{key}, norm: {norm:.2f}") # Optional: Verbose clipping log
                else:
                    clipped_grads_dict[key] = grad
            layer_grads.append(clipped_grads_dict)

        total_grad_norm = np.sqrt(total_grad_norm)
        # print(f"  Total Grad Norm (before clip): {total_grad_norm:.4f}") # Diagnostic print

        # Apply gradients using the optimizer's update method
        if self.optimizer and layer_params and layer_grads: # Ensure optimizer exists and there are grads/params
             self.optimizer.update(layer_params, layer_grads)

        # Calculate metrics
        metrics_results = {}
        for metric_fn in self.metrics:
             try:
                # Use metric name string if available, otherwise fallback
                metric_name = metric_fn.__name__ if hasattr(metric_fn, '__name__') else str(metric_fn)
                metrics_results[metric_name] = metric_fn(y_batch, predictions)
             except Exception as e:
                warnings.warn(f"Error calculating metric {metric_fn}: {e}")
                metrics_results[metric_name] = np.nan


        return loss, metrics_results
    
    def fit(self, x_train, y_train, epochs=1, batch_size=32, validation_data=None, verbose=True, callbacks_list=None):
        num_samples = x_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        if not self.built:
            self.build(input_shape=x_train.shape)

        history = {'loss': [], 'val_loss': []}
        for metric in self.metrics:
            history[metric.__name__] = []
            history[f'val_{metric.__name__}'] = []

        callbacks_list = callbacks_list if callbacks_list else []
        _callbacks = [callbacks.get_callback(cb) if isinstance(cb, str) else cb for cb in callbacks_list]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_metrics = {metric.__name__: [] for metric in self.metrics}

            # Training
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]

                loss, batch_metrics = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                for metric_name, value in batch_metrics.items():
                    epoch_metrics[metric_name].append(value)

                if verbose:
                    log_str = f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {loss:.4f}"
                    for m, v in batch_metrics.items():
                        metric_val = v 
                        log_str += f", {m}: {metric_val:.4f}"
                    print(log_str, end='\r')
            print() 

            epoch_loss /= num_batches
            history['loss'].append(epoch_loss)
            for metric in self.metrics:
                history[metric.__name__].append(np.mean(epoch_metrics[metric.__name__]))

            # Validation (if provided)
            if validation_data:
                val_loss, val_metrics = self.evaluate(validation_data[0], validation_data[1], batch_size=batch_size, verbose=0)
                history['val_loss'].append(val_loss)
                for metric_name, value in val_metrics.items():
                    history[f'val_{metric_name}'].append(value)
                if verbose:
                    log_str = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                    for metric in self.metrics:
                        log_str += f", {metric.__name__}: {history[metric.__name__][-1]:.4f}, Val {metric.__name__}: {history[f'val_{metric.__name__}'][-1]:.4f}"
                    print(log_str)

                # Callback execution
                for cb in _callbacks:
                    if hasattr(cb, '__call__'):
                        stop_training = cb(val_loss)
                        if stop_training:
                            print("Early stopping triggered.")
                            return history
                    elif hasattr(cb, 'on_epoch_end'):
                        logs = {'loss': epoch_loss, 'val_loss': val_loss}
                        logs.update({m.__name__: history[m.__name__][-1] for m in self.metrics})
                        logs.update({f'val_{m.__name__}': history[f'val_{m.__name__}'][-1] for m in self.metrics})
                        cb.on_epoch_end(epoch, logs=logs)

            else:
                if verbose:
                    log_str = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}"
                    for metric in self.metrics:
                        log_str += f", {metric.__name__}: {history[metric.__name__][-1]:.4f}"
                    print(log_str)
                for cb in _callbacks:
                    if hasattr(cb, 'on_epoch_end'):
                        logs = {'loss': epoch_loss}
                        logs.update({m.__name__: history[m.__name__][-1] for m in self.metrics})
                        cb.on_epoch_end(epoch, logs=logs)

        return history

    def predict(self, x, batch_size=32):
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        predictions = []
        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_samples)
            x_batch = x[start:end]
            preds_batch = self.forward(x_batch, training=False)
            predictions.append(preds_batch)
        return np.concatenate(predictions, axis=0)

    def evaluate(self, x_test, y_test, batch_size=32, verbose=1):
        num_samples = x_test.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        total_loss = 0
        metric_sums = {metric.__name__: 0 for metric in self.metrics}

        for batch in range(num_batches):
            
            start = batch * batch_size
            end = (batch + 1) * batch_size
            x_batch, y_batch = x_test[start:end], y_test[start:end]

            predictions = self.forward(x_batch, training=False)
            loss = self.loss_fn(y_batch, predictions)
            total_loss += loss

            for metric in self.metrics:
                metric_value = metric(y_batch, predictions)
                metric_sums[metric.__name__] += metric_value

        avg_loss = total_loss / num_batches
        avg_metrics = {name: metric_sums[name] / num_batches for name in metric_sums}

        if verbose:
            print(f"Evaluation Loss: {avg_loss:.4f}")
            for name, value in avg_metrics.items():
                print(f"{name}: {value:.4f}")

        return avg_loss, avg_metrics


    def summary(self):
        print("Model Summary:")
        print("=" * 30)
        print(f"{'Layer (type)':<20} {'Output Shape':<15} {'Param #':<10}")
        print("=" * 30)
        total_params = 0
        input_shape = None
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            output_shape = layer.compute_output_shape(input_shape) if input_shape else layer.output_shape
            params = layer.params
            num_params = sum(np.prod(p.shape) for p in params.values())
            print(f"{name:<20} {str(output_shape):<15} {num_params:<10}")
            total_params += num_params
            input_shape = output_shape
        print("=" * 30)
        print(f"Total params: {total_params}")
        print("=" * 30)

    def export(self, filepath):
        model_config = {
            'class_name': self.__class__.__name__,
            'config': {
                'layers': []
            }
        }
        for layer in self.layers:
            # --- Start Modification ---
            layer_class_name = layer.__class__.__name__
            config_params = {} # Store only necessary config

            # Example for Dense layer (adapt for others like Dropout)
            if layer_class_name == 'Dense':
                config_params['units'] = layer.units
                # Store activation function name, assuming it's stored or retrievable
                config_params['activation'] = layer.activation_name if hasattr(layer, 'activation_name') else None
                config_params['name'] = layer.name
                # Add other relevant config like initializers if needed and serializable

            elif layer_class_name == 'Dropout':
                 config_params['rate'] = layer.rate
                 config_params['name'] = layer.name

            # Add elif for other layer types...

            else:
                warnings.warn(f"Export: Don't know how to serialize config for layer type {layer_class_name}. Saving minimal info.")
                config_params['name'] = layer.name # Save name at least

            # Ensure all values in config_params are serializable
            serializable_config = {}
            for k, v in config_params.items():
                if isinstance(v, np.ndarray):
                    serializable_config[k] = v.tolist()
                elif isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    serializable_config[k] = int(v) # Convert numpy ints
                elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                     serializable_config[k] = float(v) # Convert numpy floats
                elif isinstance(v, (np.bool_)):
                     serializable_config[k] = bool(v) # Convert numpy bools
                elif isinstance(v, type): # e.g. initializer functions/classes
                     serializable_config[k] = v.__name__ # Store name if possible
                elif callable(v): # e.g. activation functions
                     serializable_config[k] = v.__name__ # Store name if possible
                else:
                     serializable_config[k] = v # Assume others are serializable

            layer_config = {
                'class_name': layer_class_name,
                'config': serializable_config, # Use the explicitly created config
                'weights': {name: w.tolist() for name, w in layer.params.items()}
            }
            # --- End Modification ---
            model_config['config']['layers'].append(layer_config)

        # Also a good idea to use .json extension for JSON files
        if not filepath.lower().endswith('.json'):
            warnings.warn(f"Exporting model to '{filepath}'. Consider using a '.json' extension for JSON format.")

        with open(filepath, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"Model exported to: {filepath}")

    @classmethod
    def load(cls, filepath, custom_objects=None):
        """
        Loads a model architecture and weights from a JSON file.

        Args:
            filepath (str): Path to the JSON file.
            custom_objects (dict, optional): Dictionary mapping names
                (strings) of custom layer classes or functions to the actual
                classes or functions themselves. Defaults to None.

        Returns:
            Model: A loaded Model instance.
        """
        custom_objects = custom_objects or {} # Ensure it's a dict

        print(f"Attempting to load model configuration from: {filepath}")
        try:
            with open(filepath, 'r') as f:
                model_config = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            raise # Re-raise the exception
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {filepath}. Invalid format? {e}")
            raise # Re-raise the exception
        except Exception as e:
            print(f"An unexpected error occurred opening or reading {filepath}: {e}")
            raise # Re-raise the exception

        # Validate top-level structure
        if 'class_name' not in model_config or 'config' not in model_config or 'layers' not in model_config.get('config', {}):
             raise ValueError(f"Invalid model config structure in {filepath}. Missing 'class_name', 'config', or 'config.layers'.")

        if model_config['class_name'] != cls.__name__:
            raise ValueError(f"File structure mismatch: Expected model class '{cls.__name__}', but found '{model_config['class_name']}' in the config.")

        layers = []
        print("Instantiating layers from configuration...")
        layer_configs = model_config['config']['layers']

        # Need access to layer classes (Dense, Dropout, etc.)
        # They should either be imported globally where Model is defined,
        # or passed reliably via custom_objects.
        available_layer_classes = globals().copy() # Get globally defined classes/functions
        if custom_objects:
            available_layer_classes.update(custom_objects) # Add/override with custom objects

        for i, layer_config in enumerate(layer_configs):
            # Validate layer config structure
            if 'class_name' not in layer_config or 'config' not in layer_config:
                 raise ValueError(f"Invalid config for layer #{i} in {filepath}. Missing 'class_name' or 'config'.")

            class_name = layer_config['class_name']
            config = layer_config['config']
            # Use .get('weights', {}) to handle cases where weights might be missing (e.g., non-trainable layers)
            weights_dict = layer_config.get('weights', {})
            weights = {name: np.array(weight) for name, weight in weights_dict.items()}

            print(f"  - Loading layer {i}: type '{class_name}' with config: {list(config.keys())}")

            # Find the layer class definition
            if class_name in available_layer_classes:
                layer_class = available_layer_classes[class_name]
            # Backward compatibility check (might be needed if older versions saved differently)
            # elif class_name in custom_objects:
            #     layer_class = custom_objects[class_name]
            else:
                # Provide more context in the error message
                raise ValueError(f"Unknown layer class '{class_name}' encountered for layer #{i}. "
                                 f"Ensure '{class_name}' is imported or passed in 'custom_objects'. "
                                 f"Available classes via globals/custom_objects: {list(available_layer_classes.keys())}")

            try:
                # Instantiate the layer using its saved configuration
                layer = layer_class(**config)
            except TypeError as e:
                 print(f"\nError instantiating layer type '{class_name}' (layer #{i}).")
                 print(f"Saved config keys: {list(config.keys())}")
                 import inspect
                 try:
                     sig = inspect.signature(layer_class.__init__)
                     print(f"Expected __init__ parameters: {list(sig.parameters.keys())}")
                 except Exception:
                     print("Could not inspect layer __init__ signature.")
                 print(f"TypeError: {e}\n")
                 raise TypeError(f"Mismatch between saved config and __init__ for layer '{class_name}'. {e}") from e
            except Exception as e:
                 print(f"\nUnexpected error during instantiation of layer type '{class_name}' (layer #{i}): {e}")
                 raise # Re-raise other instantiation errors


            # Set the loaded weights (if any)
            if weights:
                 # Check if layer has a 'params' attribute (or similar mechanism to store weights)
                 if hasattr(layer, 'params') and isinstance(layer.params, dict):
                     layer.params.update(weights)
                     print(f"    Loaded {len(weights)} weight arrays.")
                 else:
                     # If layer doesn't store weights this way, how should they be set? Needs layer design consideration.
                      print(f"    Warning: Layer type '{class_name}' has no 'params' dict attribute. Cannot load weights for layer #{i}.")


            # Layer is created and weights potentially set, add to list
            layers.append(layer)

        # Create the Model instance with the loaded layers
        model = cls(layers)
        print("Model layers instantiated.")

        # Determine the 'built' status of the loaded model
        # Use getattr for robustness, as layers might not initialize 'built' in __init__
        model.built = any(getattr(layer, 'built', False) for layer in layers)
        print(f"Model 'built' status inferred as: {model.built}")

        print("Model loading complete.")
        return model