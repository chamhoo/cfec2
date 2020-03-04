"""
auther: leechh
"""
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from time import time
from sklearn.model_selection import StratifiedKFold
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from .cv import CV
plt.style.use('ggplot')


class Tuning(object):
    def __init__(self, get_score_func, verbose=1):
        # use %matplotlib qt5 if verbose == 3
        self.get_score = get_score_func
        self.verbose = verbose
        self.hp = HP()
        self.log = TuningLog()

    def fmin(self, space_dict, max_evals=100):
        self.hp.get_space_dict(space_dict)
        # log param
        self.log.get_param_info(self.hp.PARAM_LIST(), self.hp.PARAM_TYPE())
        # hyperopt fmin
        return fmin(self.f, self.hp.HYPER_SD(), algo=tpe.suggest, max_evals=max_evals, trials=Trials())

    def f(self, params):
        t0 = time()
        score = self.get_score(params)
        t1 = time() - t0
        update = self.log.update(score, params, t1)
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


class HP(object):
    def __init__(self):
        self.__space_dict = dict()
        self.hp = hp

    def get_space_dict(self, space_dict):
        """
        :param space_dict: {'label_1': (hp.choice, (['aa', 'bb', 'cc'])),
                            'label_2': (hp.uniform, (low, high))}
                               +            +            +
                               +            +            +
                             label    distrib func  range tuple
        """
        self.__space_dict = space_dict

    def HYPER_SD(self):
        sd = {}
        for label, (func, params) in self.__space_dict.items():
            if func.__name__ == 'hp_choice':
                sd[label] = func(label, params)
            else:
                sd[label] = func(label, *params)
        return sd

    def PARAM_LIST(self):
        return list(self.__space_dict.keys())

    def PARAM_TYPE(self):
        pt = {}
        for label, (func, params) in self.__space_dict.items():
            if func.__name__ in ['hp_choice', 'hp_randint']:
                pt[label] = 'discrete'
            elif func.__name__ in ['hp_loguniform', 'hp_qloguniform', 'hp_lognormal', 'hp_qlognormal']:
                pt[label] = 'log'
            else:
                pt[label] = 'numbers'
        return pt

    @staticmethod
    def help():
        print('- https://github.com/hyperopt/hyperopt/wiki/FMin')


class TuningLog(object):
    def __init__(self):
        self.best_score = 2**32
        self.best_param = {}

        self.eval = 0
        self.score = []
        self.isupdate = []
        self.usetime = []
        self.param = dict()

        self.param_lst = None
        self.param_type = None

    def get_param_info(self, param_lst, param_type):
        if ((self.param_lst is not None) & (self.param_lst != self.param_lst)) or \
                ((self.param_type is not None) & (self.param_type != self.param_type)):
            raise ValueError(f'you need keep this two values when the older is not None.')
        else:
            self.param_lst = param_lst
            self.param_type = param_type

    def update(self, score, param, times):
        if score < self.best_score:
            update = True
            self.best_param = param
            self.best_score = score
        else:
            update = False

        self.eval += 1
        self.usetime.append(times)
        self.score.append(score)
        self.isupdate.append(update)
        # param
        for key, val in param.items():
            if self.eval == 1:
                self.param[key] = [val]
            else:
                self.param[key].append(val)
        return update

    def get_eval(self):
        return self.eval

    def get_log(self):
        log_df = {'score': self.score, 'update': self.isupdate, 'usetime': self.usetime}
        for key, val in self.param.items():
            log_df[key] = val
        return pd.DataFrame(log_df)

    def get_best_score(self):
        return self.best_score

    def get_best_param(self):
        return self.best_param

    def plot(self, score_interval=None):
        loc = self.cal_loc()
        plt.figure(figsize=[loc[1] * 7, loc[0] * 7])
        self.plotcomp(False, score_interval)

    def plotcomp(self, pause=False, score_interval=None):
        loc = self.cal_loc()
        plt.clf()
        for idx, param in enumerate(self.param_lst):
            plt.subplot(loc[0], loc[1], idx + 1)
            # numbers
            if self.param_type[param] == 'numbers':
                plt.scatter(x=self.param[param], y=self.score, c='salmon', alpha=0.8)
                plt.xlabel(param)
            # log
            elif self.param_type[param] == 'log':
                log_x = [np.log(x) for x in self.param[param]]
                plt.scatter(x=log_x, y=self.score, c='salmon', alpha=0.8)
                plt.xlabel(f'np.log({param})')
            # discrete
            elif self.param_type[param] == 'discrete':
                df = pd.DataFrame({'score': self.score, param: self.param[param]})
                data = [df.score[df[param] == element].values for element in df[param].unique()]
                plt.violinplot(data, showmeans=True, showmedians=True)
                plt.xticks([i + 1 for i in range(len(data))], df[param].unique())
                plt.xlabel(param)
            # title & labels
            plt.ylabel('score')
            if score_interval is not None:
                print(score_interval)
                plt.ylim(*score_interval)
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
