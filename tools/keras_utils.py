"""
auther: leechh
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1)) \
               - K.mean((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def gelu(inputs):
    return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))