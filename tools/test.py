import numpy as np
from time import sleep
from hyperopt import hp
from .auto_tunning import Tuning


def scorefunc(params):
    if params['p_choice']:
        score = (params['p_uniform'] + np.log(params['p_log']) + params['p_normal']) * 2
    else:
        score = params['p_uniform'] + np.log(params['p_log']) + params['p_normal']
    sleep(0.1)
    return score


sd = {
    'p_choice': (hp.choice, (True, False)),
    'p_uniform': (hp.uniform, (0, 2)),
    'p_log': (hp.loguniform, (-10, 1)),
    'p_normal': (hp.normal, (2, 1))
}

tu = Tuning(scorefunc, 3)
tu.fmin(sd, 100)
