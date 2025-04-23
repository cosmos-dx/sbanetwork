"""
callbacks.py
------------

This module provides a set of training callbacks for the SBA Network (Sparse Biological-inspired Adaptive Network) library.  
Callbacks allow custom behavior to be applied during training, such as stopping early when no improvement is detected, saving the best model, or adjusting the learning rate dynamically.

Classes:
    - EarlyStopping: Stops training when the validation loss stops improving.
    - ModelCheckpoint: Saves the model at its best performance during training.
    - LearningRateScheduler: Adjusts the learning rate based on a user-defined schedule.

Functions:
    - get_callback(name, **kwargs): Dynamically retrieves a callback instance by name with specified parameters.

Dictionary:
    - callbacks: A registry of available callback class references for flexible instantiation.

Notes
-----
Author: Abhishek Gupta  
Library: sbanetwork 
Email: abhishekgupta0118@gmail.com
Github: cosmos-dx 

The `sbanetwork` library emphasizes modular, flexible, and biologically inspired neural network construction.  
The `callbacks.py` module offers essential training-time utilities for better control, adaptability, and efficiency during learning. It supports regression and classification use-cases and integrates seamlessly with custom training loops.
"""



import numpy as np
import os
import pickle
# Each callback class implements a method to be called at the end of each epoch during training.
# The callbacks are designed to be used with regression and classification tasks.
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.Inf
        self.counter = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = np.Inf

    def save(self, model, current_value):
        if not self.save_best_only or current_value < self.best:
            self.best = current_value
            with open(self.filepath, 'wb') as f:
                pickle.dump(model, f)

class LearningRateScheduler:
    def __init__(self, schedule):
        self.schedule = schedule

    def get_lr(self, epoch):
        return self.schedule(epoch)


callbacks = {
    'early_stopping': EarlyStopping,
    'model_checkpoint': ModelCheckpoint,
    'learning_rate_scheduler': LearningRateScheduler,
   
}

def get_callback(name, **kwargs):
    
    name = name.lower()
    if name not in callbacks:
        raise ValueError(f"Initializer '{name}' is not supported. Available options are: {list(callbacks.keys())}")
    return callbacks[name](**kwargs)
