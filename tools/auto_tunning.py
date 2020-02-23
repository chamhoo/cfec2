"""
auther: leechh
"""
import sklearn
import numpy as np
import pandas as pd
from time import sleep, time
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from .cv import CV


class Tuning(object):
    def __init__(self, get_score_func, verbose=1):
        # use %matplotlib qt5 if verbose == 3
        self.get_score = get_score_func
        self.verbose = verbose
        self.hp = hp
        self.log = TuningLog()

    def fmin(self, space_dict, max_evals=100):
        return fmin(self.f, space_dict, algo=tpe.suggest, max_evals=max_evals, trials=Trials())

    def f(self, params):
        score = self.get_score(params)
        update = self.log.update(score, params)
        if self.verbose == 3:
            if self.log.get_eval() == 1:
                loc = self.log.cal_loc()
                plt.figure(figsize=[loc[1] * 7, loc[0] * 7])
                plt.ion()
            # plt
            self.log.plotcomp(True)
        else:
            self.show(score, params, update)
        return {'loss': score, 'status': STATUS_OK}

    def show(self, score, param, update):
        if (self.verbose == 1) & update:
            print(f'new best, eval {self.log.get_eval()}, score {round(score, 4)}, param {param}')
        if self.verbose == 2:
            print(f'eval {self.log.get_eval()}, score {round(score, 4)}, param {param}')


class TuningLog(object):
    def __init__(self):
        self.best_score = 2**32
        self.best_param = {}

        self.eval = 0
        self.score = []
        self.isupdate = []
        self.param = dict()
        self.param_lst = []
        self.param_type = dict()

    def update(self, score, param):
        if score < self.best_score:
            update = True
            self.best_param = param
            self.best_score = score
        else:
            update = False

        self.eval += 1
        self.score.append(score)
        self.isupdate.append(update)
        # param
        for key, val in param.items():
            if self.eval == 1:
                self.param_lst = list(param.keys())
                self.param[key] = [val]
                if isinstance(val, (int, float)):
                    self.param_type[key] = 'numbers'
                else:
                    self.param_type[key] = 'str'
            else:
                self.param[key].append(val)
        return update

    def get_eval(self):
        return self.eval

    def get_log(self):
        log_df = {'score': self.score, 'update': self.isupdate}
        for key, val in self.param.items():
            log_df[key] = val
        return pd.DataFrame(log_df)

    def get_best_score(self):
        return self.best_score

    def get_best_param(self):
        return self.best_param

    def plot(self):
        loc = self.cal_loc()
        plt.figure(figsize=[loc[1] * 7, loc[0] * 7])
        self.plotcomp()

    def plotcomp(self, pause=False):
        loc = self.cal_loc()
        plt.cla()
        for idx, param in enumerate(self.param_lst):
            plt.subplot(loc[0], loc[1], idx + 1)
            if self.param_type[param] == 'numbers':
                plt.scatter(x=self.param[param], y=self.score, c='salmon', alpha=0.8)
            elif self.param_type[param] == 'str':
                df = pd.DataFrame({'score': self.score, param: self.param[param]})
                data = [df.score[df[param] == element].values for element in df[param].unique()]
                plt.violinplot(data, showmeans=True, showmedians=True)
                plt.xticks([i + 1 for i in range(len(data))], df[param].unique())
            plt.title(param)
            plt.xlabel(param)
            plt.ylabel('score')
        if pause:
            plt.pause(0.001)
        plt.show()

    def cal_loc(self):
        len_p = len(self.param_lst)
        loc1 = int(np.ceil(np.sqrt(len_p)))
        loc0 = int(np.ceil(len_p / loc1))
        return [loc0, loc1]


class CVGetScore(object):

    def __init__(self, x, y, metrics_func, split_method, nfolds=5, random_state=2333,
                 model=None, cv_fit_params=None, model_fix_params=None, model_search_space=None):
        self.x = x
        self.y = y
        self.model = model
        self.metrics_func = metrics_func
        self.split_method = split_method
        self.nfolds = nfolds
        self.random_state = random_state
        self.cv_fit_params = cv_fit_params
        self.model_fix_params = model_fix_params
        self.model_search_space = model_search_space

    def __call__(self, params):
        # cv param
        cvparam = dict()
        cvparam['nfolds'] = self.nfolds
        cvparam['random_state'] = self.random_state

        model_param = dict()
        for key, val in params.items():
            model_param[key] = val
        for key, val in self.model_fix_params.items():
            model_param[key] = val
        cvparam['model'] = self.model(**model_param)

        # fit param
        fit_param = dict()
        fit_param['x'] = self.x
        fit_param['y'] = self.y
        fit_param['metrics_func'] = self.metrics_func
        fit_param['split_method'] = self.split_method
        for key, val in self.cv_fit_params.items():
            fit_param[key] = val
        cv = CV(**cvparam)
        score = cv.fit(**fit_param)
        K.clear_session()
        cv.clear_model()
        del cvparam
        return score

    def autoload(self, modelname):
        p_dict = self.params_dict()
        self.model, self.cv_fit_params, self.model_fix_params, self.model_search_space = p_dict[modelname]

    def params_dict(self):
        prm_dict = {
            'LogisticRegression': [
                sklearn.linear_model.LogisticRegression,
                {
                    'fit_params': None,
                    'eval_param': None,
                    'use_proba': True,
                    'verbose': False
                },
                {
                    'penalty': 'l2',
                    'random_state': 2333,
                    'max_iter': 10000,
                    'n_jobs': -1
                },
                {
                    'C': hp.loguniform('C', -10, 0),
                    'solver': hp.choice('solver', ['liblinear', 'sag', 'saga'])
                }
            ]
        }
        return prm_dict

    def GET_SEARCH_SPACE(self):
        return self.model_search_space
