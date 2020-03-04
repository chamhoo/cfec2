"""
auther: leechh
"""
import math
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class CyclicLR(callbacks.Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        K.set_value(self.model.optimizer.lr, self.clr())


class MaxLrFinder(callbacks.Callback):
    def __init__(self, min_lr, max_lr, epochs, batch_size, sample_length):
        """
        When using this callback, do not use tf.keras.callbacks.EarlyStopping or
         any operation that will call self.model.stop_training
        :param min_lr:
        :param max_lr:
        :param epochs:
        :param batch_size:
        :param sample_length:
        """
        super(MaxLrFinder).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.sample_length = sample_length

        self.result = []
        self.result_name_list = []
        self.num_model = 0
        self.batch_id = []
        self.rate = (max_lr/min_lr)**(1/(epochs*sample_length/batch_size))

    def on_train_begin(self, logs=None):
        self.result.append([])

        self.batch_id.append(0)
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_train_end(self, logs=None):
        self.num_model += 1

    def on_batch_begin(self, batch, logs=None):
        new_lr = self.min_lr * self.rate**self.batch_id[self.num_model]
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_batch_end(self, batch, logs=None):
        logk = list(logs.keys())
        logk.remove('batch')
        logk.remove('size')

        if self.batch_id[self.num_model] == 0:
            self.result_name_list = logk

        log_lst = []
        for k in self.result_name_list:
            log_lst.append(logs[k])

        self.batch_id[self.num_model] += 1
        self.result[self.num_model].append(log_lst)

    def get_result(self):
        result_arr = np.array(self.result)   # (num_model, num_batch_per_model, n_Metric)
        if np.ndim(result_arr) != 3:
            raise ValueError('result_arr has a wrong shape, the possible reason is '
                             'that each model has a different number of batches')
        result_dict = dict(zip(self.result_name_list, np.rollaxis(result_arr, 2).tolist()))
        return result_dict

    def plot(self, show_exp_lr=False):
        result_dict = self.get_result()
        lr_func = lambda t: self.min_lr * self.rate ** t
        lr_list = [math.log(lr_func(t), 0.1) if show_exp_lr else lr_func(t) for t in range(self.batch_id[0])]
        # loc
        len_p = len(self.result_name_list)
        loc1 = int(np.ceil(np.sqrt(len_p)))
        loc0 = int(np.ceil(len_p / loc1))

        plt.figure(figsize=[loc1 * 7, loc0 * 7])
        for idx, (key, val) in enumerate(result_dict.items()):
            plt.subplot(loc0, loc1, idx + 1)
            for sub_res in val:
                plt.plot(lr_list, sub_res)
                plt.xlabel('lr')
                plt.ylabel(key)
