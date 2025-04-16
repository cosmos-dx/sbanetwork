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
