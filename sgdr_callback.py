import numpy as np
from keras.callbacks import Callback
from keras import backend as K


class SGDR(Callback):
    """This callback implements the learning rate schedule for
    Stochastic Gradient Descent with warm Restarts (SGDR),
    as proposed by Loshchilov & Hutter (https://arxiv.org/abs/1608.03983).
    
    The learning rate at each epoch is computed as:
    lr(i) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * i/num_epochs))
    
    Here, num_epochs is the number of epochs in the current cycle, which starts
    with base_epochs initially and is multiplied by mul_epochs after each cycle.
    
    # Example
        ```python
            sgdr = CyclicLR(min_lr=0.0, max_lr=0.05,
                                base_epochs=10, mul_epochs=2)
            model.compile(optimizer=keras.optimizers.SGD(decay=1e-4, momentum=0.9),
                          loss=loss)
            model.fit(X_train, Y_train, callbacks=[sgdr])
        ```
    
    # Arguments
        min_lr: minimum learning rate reached at the end of each cycle.
        max_lr: maximum learning rate used at the beginning of each cycle.
        base_epochs: number of epochs in the first cycle.
        mul_epochs: factor with which the number of epochs is multiplied
                after each cycle.
    """

    def __init__(self, min_lr=0.0, max_lr=0.05, base_epochs=10, mul_epochs=2):
        super(SGDR, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.base_epochs = base_epochs
        self.mul_epochs = mul_epochs

        self.cycles = 0.
        self.cycle_iterations = 0.
        self.trn_iterations = 0.

        self._reset()

    def _reset(self, new_min_lr=None, new_max_lr=None,
               new_base_epochs=None, new_mul_epochs=None):
        """Resets cycle iterations."""
        
        if new_min_lr != None:
            self.min_lr = new_min_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_base_epochs != None:
            self.base_epochs = new_base_epochs
        if new_mul_epochs != None:
            self.mul_epochs = new_mul_epochs
        self.cycles = 0.
        self.cycle_iterations = 0.
        
    def sgdr(self):
        
        cycle_epochs = self.base_epochs * (self.mul_epochs ** self.cycles)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * (self.cycle_iterations + 1) / cycle_epochs))
        
    def on_train_begin(self, logs=None):
        
        if self.cycle_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.sgdr())
            
    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        
        self.trn_iterations += 1
        self.cycle_iterations += 1
        if self.cycle_iterations >= self.base_epochs * (self.mul_epochs ** self.cycles):
            self.cycles += 1
            self.cycle_iterations = 0
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.sgdr())
