from keras.callbacks import Callback


class LossHistory(Callback):
    """Custom callback function to track training loss and validation loss

    Params:
        test_data: tuple of (X_test, y_test)
        interval: Intervals at which validation has to be carried out
    """

    def __init__(self, test_data, interval=10):
        super(LossHistory, self).__init__()
        self.seen = 0
        self.interval = interval
        self.test_data = test_data

    def on_train_begin(self, logs=None):
        self.train_indices, self.train_losses, self.test_indices, self.test_losses, self.test_acc = [], [], [], [], []

    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        if self.seen % self.interval == 0:
            val_loss, val_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
            self.test_losses.append(val_loss)
            self.test_indices.append(self.seen)
            self.test_acc.append(val_acc)
        self.train_losses.append(logs.get('loss'))
        self.train_indices.append(self.seen)
