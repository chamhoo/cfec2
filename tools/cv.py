"""
auther: leechh
"""
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.externals import joblib
from glob import glob
from copy import deepcopy


class CV(object):
    def __init__(self, model, nfolds=5, random_state=23333):
        self.score = None
        self.feature_importance = {}
        self.random_state = random_state
        self.model = self.__model_compliance(model, nfolds)

    @staticmethod
    def __model_compliance(model, nfolds=5):
        model_lst = list()
        if isinstance(model, tf.keras.Model):
            for _ in range(nfolds):
                new_model = tf.keras.models.clone_model(model)
                new_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
                new_model.set_weights(model.get_weights())
                model_lst.append(new_model)

        elif isinstance(model, list):
            if len(model) == nfolds:
                model_lst = model
            else:
                raise IndexError(f'len(model) should be same as nfolds and should be {nfolds}, not {len(model)}')

        else:
            if callable(getattr(model, 'fit')) and callable(getattr(model, 'predict')):
                model_lst = [deepcopy(model) for _ in range(nfolds)]
            else:
                AttributeError('model class should have 2 basic Attribute: -fit, -predict')
        return model_lst

    def __len__(self):
        return len(self.model)

    def reset_model(self, model_lst):
        if len(model_lst) != self.__len__():
            raise IndexError(f'len(model_lst) is {self.__len__()}, not {len(model_lst)}')
        self.model = model_lst

    def fit(self, x, y, metrics_func, split_method=None,
            fit_params=None, eval_param=None, use_proba=False, verbose=False, fit_use_valid=False,
            output_oof_pred=False):
        # x, y
        x_is_list = isinstance(x, list)
        y_is_list = isinstance(y, list)
        x_is_df = isinstance(x, pd.DataFrame)
        
        # oof_pred
        oof_pred = np.zeros([len(y), 1]) if output_oof_pred else None

        # params
        if split_method is None:
            split_method = ShuffleSplit if self.__len__() == 1 else KFold

        if self.__len__() == 1:
            split_method = split_method(n_splits=1,
                                        random_state=self.random_state,
                                        test_size=0.2)
        else:
            split_method = split_method(n_splits=self.__len__(),
                                        shuffle=True,
                                        random_state=self.random_state)

        fit_params = {} if fit_params is None else fit_params
        eval_param = {} if eval_param is None else eval_param

        # split & train
        score_lst = list()
        X = x[0] if x_is_list else x
        for idx, (train_idx, valid_idx) in enumerate(split_method.split(X, y)):
            # x
            if x_is_list:
                x_train = [data[train_idx] for data in x]
                x_valid = [data[valid_idx] for data in x]
            elif x_is_df:
                x_train = x.loc[train_idx]
                x_valid = x.loc[valid_idx]
            else:
                x_train = x[train_idx]
                x_valid = x[valid_idx]
            # y
            if y_is_list:
                y_train = [data[train_idx] for data in y]
                y_valid = [data[valid_idx] for data in y]
            else:
                y_train = y[train_idx]
                y_valid = y[valid_idx]
            # fit
            if fit_use_valid:
                fit_params['validation_data'] = (x_valid, y_valid)
            self.model[idx].fit(x_train, y_train, **fit_params)
            # predict
            y_pred = self.__pred(self.model[idx], x_valid, eval_param, use_proba)
            score = metrics_func(y_valid, y_pred)
            score_lst.append(score)
            if verbose:
                print(f'folds {idx} is done, score is {score}')
            if output_oof_pred:
                oof_pred[valid_idx] = y_pred

        if output_oof_pred:
            return np.mean(score_lst), oof_pred
        else:
            return np.mean(score_lst)

    def clear_model(self):
        del self.model

    def __pred(self, model, x, pred_param=None, use_proba=False):
        pred_param = {} if pred_param is None else pred_param
        if use_proba:
            y_pred = model.predict_proba(x, **pred_param)
            y_pred = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred
        else:
            y_pred = model.predict(x, **pred_param)
        return y_pred

    def predict(self, x, pred_param=None, use_proba=False):
        pred_lst = list()
        for model in self.model:
            pred_lst.append(self.__pred(model, x, pred_param, use_proba))
        return np.mean(pred_lst, axis=0)

    @staticmethod
    def mkdir(model_dir):
        if os.path.isdir(model_dir):
            isdrop = input(f'dirpath {model_dir} is exist, do you want to drop it? [Yes/No]')
            if isdrop in ['Yes', 'yes', 'y', 'Y', True]:
                shutil.rmtree(model_dir)
            if isdrop in ['No', 'no', 'n', 'N', False]:
                raise FileExistsError(f'{model_dir} is exist')
        os.mkdir(model_dir)

    def save(self, suffix, save_dir):
        self.mkdir(save_dir)
        for i, model in enumerate(self.model):
            Saver(os.path.join(save_dir, f'{i}.{suffix}'), model).save()

    def load(self, load_dir):
        load_path = []
        for suffix in Saver.get_ext_lst():
            load_path.extend(glob(os.path.join(load_dir, f'*{suffix}')))

        # raise if multiple suffixes appear
        load_path_suffix = set(os.path.splitext(i)[-1] for i in load_path)
        if len(load_path_suffix) == 0:
            raise FileNotFoundError(f'not find model file in {load_dir}, pls check again')

        elif len(load_path_suffix) > 1:
            raise MultiSuffixError

        else:
            load_path_suffix = list(load_path_suffix)[0]

        # is match
        if len(load_path) == 1:
            self.model = [Saver(load_path[0], self.model).load() for i in range(self.__len__())]

        elif len(load_path) == self.__len__():
            for idx, (model, path) in enumerate(zip(self.model, load_path)):
                self.model[idx] = Saver(path, model)

        else:
            raise IndexError('Local file has different lengths')


class Saver(object):
    def __init__(self, path, model=None):
        self.model = model
        self.path = path
        self.suffix = os.path.splitext(path)[-1]
        self.suffix_avl_lst = self.get_ext_lst()

    @staticmethod
    def get_ext_lst():
        return ['.pkl', '.cbm', 'h5']

    def save(self):
        if self.suffix == '.pkl':
            joblib.dump(self.model, self.path)

        elif self.suffix == '.cbm':
            self.model.save_model(self.path)

        elif self.suffix == 'h5':
            self.model.save(self.path)

        else:
            raise ValueError(f'{self.suffix} is not available')

    def load(self):
        if self.suffix == '.pkl':
            return joblib.load(self.path)

        elif self.suffix == '.cbm':
            self.model.load_model(self.path)
            return self.model

        elif self.suffix == 'h5':
            return self.model.load(self.path)

        else:
            raise ValueError(f'{self.suffix} is not available')


class MultiSuffixError(Exception):
    def __init__(self, err=None):
        err = 'There is more than one suffix available in this folder.' if err is None else err
        Exception.__init__(err)