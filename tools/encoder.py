"""
auther: leechh
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


class IdxValEncoder(object):
    def __init__(self, feature_col, bin_col=None,
                 class_col=None, num_col=None):
        # cols
        self.feature_col = feature_col
        self.bin_col = [] if bin_col is None else bin_col
        self.class_col = [] if class_col is None else class_col
        self.num_col = [] if num_col is None else num_col
        # feature_dict
        self.first_fit = True
        self.feature_idx = 0
        self.feature_dict = dict()
        for feature in self.feature_col:
            self.feature_dict[feature] = dict()

    @staticmethod
    def check_parameter(x, feature_cols):
        # x
        if isinstance(x, list):
            x_iter = x
        elif isinstance(x, np.ndarray):
            x_iter = x
        elif isinstance(x, pd.DataFrame):
            x_iter = x.itertuples(index=False)
            feature_cols = x.columns
        else:
            raise TypeError(f'x_iter type should in [list, numpy.ndarray, '
                            f'pandas.DataFrame] NOT {type(x)}')
        # feature_cols
        if feature_cols is None:
            raise ValueError('feature_col is not None')
        return x_iter, list(feature_cols)

    def fit(self, x, feature_cols=None, verbose=0):
        # check
        x_iter, feature_cols = self.check_parameter(x, feature_cols)
        # iter
        generator = tqdm(enumerate(x_iter)) if verbose else enumerate(x_iter)
        for idx, col in generator:
            for feature_name, val in zip(feature_cols, col):

                if (idx == 1) & (self.first_fit is True) &\
                        ((feature_name in self.bin_col) or (feature_name in self.num_col)):
                    self.feature_dict[feature_name][feature_name] = self.feature_idx
                    self.feature_idx += 1

                elif (feature_name in self.class_col) & \
                        (self.feature_dict[feature_name].__contains__(val) is False):
                    self.feature_dict[feature_name][val] = self.feature_idx
                    self.feature_idx += 1
        self.first_fit = False

    def transform(self, x, feature_cols=None, verbose=0):
        # idx & val
        idx_list = []
        val_list = []
        # check
        x_iter, feature_cols = self.check_parameter(x, feature_cols)
        # iter
        generator = tqdm(enumerate(x_iter)) if verbose else enumerate(x_iter)
        for idx, col in generator:
            col_idx, col_val = [], []
            for feature_name, val in zip(feature_cols, col):
                if (feature_name in self.bin_col) or (feature_name in self.num_col):
                    col_idx.append(self.feature_dict[feature_name][feature_name])
                    col_val.append(float(val))
                elif feature_name in self.class_col:
                    col_idx.append(self.feature_dict[feature_name][val])
                    col_val.append(1)
            idx_list.append(col_idx)
            val_list.append(col_val)
        return np.array(idx_list), np.array(val_list)

    def fit_transform(self, x, feature_cols=None, verbose=0):
        self.fit(x, feature_cols, verbose)
        return self.transform(x, feature_cols, verbose)

    def get_vocabulary(self):
        return self.feature_idx + 1
