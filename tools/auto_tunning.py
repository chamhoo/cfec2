"""
auther: leechh
"""
import os
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from glob import glob


class Tuning(object):
    def __init__(self, X, y, model, model_fix_params, cv_params, metrics_func=None, verbose=1, fit_use_metrics=False):
        self.x = X
        self.y = y
        self.model = model
        self.model_fix_params = model_fix_params
        self.cv_params = cv_params
        self.metrics_func = metrics_func
        self.verbose = verbose
        self.fit_use_metrics = fit_use_metrics

        self.eval = 0
        self.hp = hp
        self.best_param = {}
        self.best_score = 2**32   # lower is better.
        self.x_is_df = (type(self.x) == pd.core.frame.DataFrame)

    def fmin(self, space_dict, max_evals=100):
        return fmin(self.f, space_dict, algo=tpe.suggest, max_evals=max_evals, trials=Trials())

    def f(self, params):
        self.eval += 1
        update = False
        score, param = self.get_score(params)
        
        if self.best_score > score:
            self.best_score = score
            self.best_param = param
            update = True
        
        if (self.verbose == 1) & update:
            print(f'new best, eval {self.eval}, score {round(score, 4)}, param {params}')
        if self.verbose == 2:
            print(f'eval {self.eval}, score {round(score, 4)}, param {params}')
        return {'loss': score, 'status': STATUS_OK}

    def get_score(self, params):
        # params
        param = {}
        for key, value in self.model_fix_params.items():
            param[key] = value
        for key, value in params.items():
            param[key] = value
        return self.cv_score(metrics_func=self.metrics_func, model_param=param, **self.cv_params), param
    
    def __split(self, nflod, split_method, shuffle=True, random_state=23333):
        if nflod == 1:
            yield train_test_split(self.x, self.y, test_size = 0.2, random_state=random_state, stratify =self.y)
        else:
            cv = split_method(n_splits=nflod, shuffle=shuffle, random_state=random_state)
            for train_idx, valid_idx in cv.split(self.x, self.y):
                
                x_train = self.x.loc[train_idx] if self.x_is_df else self.x[train_idx]
                x_valid = self.x.loc[valid_idx] if self.x_is_df else self.x[valid_idx]
                y_train = self.y[train_idx]
                y_valid = self.y[valid_idx]
                yield x_train, x_valid, y_train, y_valid

    def __cv(self, metrics_func, model_param, split_method, nflod=5, shuffle=True, fit_params=None, random_state=23333):
        # 返回模型和valid得分
        fit_params = {} if fit_params is None else fit_params
        for x_train, x_valid, y_train, y_valid in self.__split(nflod, split_method, shuffle, random_state):
            # model
            if self.fit_use_metrics:
                fit_params['eval_set'] = (x_valid, y_valid)
                
            model = self.model(**model_param)
            model.fit(x_train, y_train, **fit_params)
            y_predict = model.predict_proba(x_valid)[:, 1]
            score = metrics_func(y_valid, y_predict)
            yield model, score

    def cv_score(self, **kwargs):
        score_lst = []
        for model, score in self.__cv(**kwargs):
            score_lst.append(score)
        return float(np.mean(score_lst))

    def cv_predict(self, x, add_model_param=None):
        predict_lst = []
        model_param = self.best_param
        if add_model_param:
            for key, val in add_model_param.items():
                model_param[key] = val
        for idx, (model, _) in enumerate(self.__cv(model_param, **self.cv_params)):
            y_predict = model.predict_proba(x)[:, 1]
            predict_lst.append(y_predict)
        return np.mean(predict_lst, axis=0)

    
class CV(object):
    def __init__(self, random_state=23333):
        self.random_state = random_state
        self.score = None
        self.model_lst = []
        self.feature_importance = {}

    def fit(self, x, y, model, model_param, fit_params,
            metrics_func, split_method, nflods=5, shuffle=True):
        pass

    def mkdir(self, model_dir):
        if os.path.isdir(model_dir):
            isdrop = input(f'dirpath {model_dir} is exist, do you want to drop it? [Yes/No]')
            if isdrop == 'No':
                raise FileExistsError(f'{model_dir} is exist')
        os.mkdir(model_dir)

    def save_cat(self, save_dir):
        self.mkdir(save_dir)
        for i, model in enumerate(self.model_lst):
            model.save_model(os.path.join(save_dir, f'{i}.cbm'))

    def load(self, load_dir):
        self.model_lst = []
        load_path = glob(os.path.join(load_dir, '*.*'))
        for path in load_path:
            model = 
            self.model_lst.append()
