"""
auther: leechh
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, regularizers, activations
from tensorflow.keras.layers import Dense, Activation, Embedding, add, Layer, multiply


def linear_regression(vocabulary_size, feature_number,
                      activation, loss, metrics, optimizer, l1=0., l2=0.):
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    em = Embedding(vocabulary_size, 1, input_length=feature_number, name='em')(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    x = K.sum(multiply([em, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)
    # add_bias
    w0 = K.variable(K.zeros(shape=[1], dtype='float32'), name='w0')
    x = K.bias_add(x, bias=w0, data_format='channels_last')
    # regularizer
    regularizers.L1L2(l1, l2)(x)
    # activation
    act = Activation(activation) if isinstance(activation, str) else activation
    y = act(x)
    # output
    model = Model(inputs=[idx, val], outputs=y)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def fm(vocabulary_size, feature_number,
       activation, loss, metrics, optimizer, l1=0., l2=0., k=10):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # linear
    em_l = Embedding(vocabulary_size, 1, input_length=feature_number, name='em_l')(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    linear = K.sum(multiply([em_l, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)
    # pair
    em_p = Embedding(vocabulary_size, k, input_length=feature_number, name='em_p')(idx)  # (batch, feature_number, k)
    pow_multiply = K.sum(multiply([K.pow(em_p, 2), K.pow(val_ed, 2)]), 1)  # (batch_size ,k)
    multiply_pow = K.pow(K.sum(multiply([em_p, val_ed]), 1), 2)  # (batch_size, k)
    pair = 0.5 * K.sum((multiply_pow - pow_multiply), axis=1, keepdims=True)  # (batch_size, 1)
    x = add([linear, pair])
    # add_bias
    w0 = K.variable(K.zeros(shape=[1], dtype='float32'), name='w0')
    x = K.bias_add(x, bias=w0, data_format='channels_last')
    # regularizer
    regularizers.L1L2(l1, l2)(x)
    # activation
    act = Activation(activation) if isinstance(activation, str) else activation
    y = act(x)
    # output
    model = Model(inputs=[idx, val], outputs=y)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model