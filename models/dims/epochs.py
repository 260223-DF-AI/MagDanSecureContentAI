
class EpochDim(self):
        def __init__(self, epoch_key, epoch_number, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate, batch_size, is_improved):
            self.epoch_key = int(epoch_key)
            self.epoch_number = int(epoch_number)
            self.train_loss = float(train_loss)
            self.val_loss = float(val_loss)
            self.train_accuracy = float(train_accuracy)
            self.val_accuracy = float(val_accuracy)
            self.learning_rate = float(learning_rate)
            self.batch_size = int(batch_size)
            self.is_improved = bool(is_improved)


