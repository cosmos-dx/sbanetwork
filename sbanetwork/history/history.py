
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def log(self, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def get(self):
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

    def reset(self):
        self.train_losses = []
        self.val_losses = []