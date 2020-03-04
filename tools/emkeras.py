"""
auther: leechh
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, regularizers, callbacks
from tensorflow.keras.layers import Dense, Activation, Embedding, add, multiply, Dropout, BatchNormalization, Conv1D


def linear_regression(vocabulary_size, feature_number,
                      activation, loss, metrics, optimizer, l1=0., l2=0.):
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    em = Embedding(vocabulary_size, 1,
                   input_length=feature_number,
                   name='em',
                   embeddings_regularizer=regularizers.L1L2(l1, l2))(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    x = K.sum(multiply([em, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)
    # add_bias
    w0 = K.variable(K.zeros(shape=[1], dtype='float32'), name='w0')
    x = K.bias_add(x, bias=w0, data_format='channels_last')
    # activation
    act = Activation(activation) if isinstance(activation, str) else activation
    y = act(x)
    # output
    model = Model(inputs=[idx, val], outputs=y)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def fm(vocabulary_size, feature_number,
       activation, loss, metrics, optimizer, l1_linear=0., l2_linear=0., l1_pair=0., l2_pair=0., k=10):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # linear
    em_l = Embedding(vocabulary_size, 1,
                     input_length=feature_number,
                     name='em_l',
                     embeddings_regularizer=regularizers.L1L2(l1_linear, l2_linear))(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    linear = K.sum(multiply([em_l, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)

    # pair
    em_p = Embedding(vocabulary_size, k,
                     input_length=feature_number,
                     name='em_p',
                     embeddings_regularizer=regularizers.L1L2(l1_pair, l2_pair))(idx)  # (batch, feature_number, k)
    pow_multiply = K.sum(multiply([K.pow(em_p, 2), K.pow(val_ed, 2)]), 1)  # (batch_size ,k)
    multiply_pow = K.pow(K.sum(multiply([em_p, val_ed]), 1), 2)  # (batch_size, k)
    pair = 0.5 * K.sum((multiply_pow - pow_multiply), axis=1, keepdims=True)  # (batch_size, 1)
    x = add([linear, pair])
    # add_bias
    w0 = K.variable(K.zeros(shape=[1], dtype='float32'), name='w0')
    x = K.bias_add(x, bias=w0, data_format='channels_last')
    # activation
    act = Activation(activation) if isinstance(activation, str) else activation
    y = act(x)
    # output
    model = Model(inputs=[idx, val], outputs=y)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def deepfm(vocabulary_size, feature_number,
           activation, loss, metrics, optimizer,
           l1_linear=0., l2_linear=0.,
           l1_pair=0., l2_pair=0., use_fm=True,
           l1_deep=0., l2_deep=0., use_deep=True,
           deep_dropout=0., deep_use_bn=False,
           num_deep_layer=2, num_neuron=128, deep_activation='relu', k=5):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # linear
    em_l = Embedding(vocabulary_size, 1,
                     input_length=feature_number,
                     name='em_l',
                     embeddings_regularizer=regularizers.L1L2(l1_linear, l2_linear))(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    linear = K.sum(multiply([em_l, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)

    # pair
    em_p = Embedding(vocabulary_size, k,
                     input_length=feature_number,
                     name='em_p',
                     embeddings_regularizer=regularizers.L1L2(l1_pair, l2_pair))(idx)  # (batch, feature_number, k)
    em = multiply([em_p, val_ed])    # (batch, feature_number, k)
    pow_multiply = K.sum((K.pow(em, 2)), 1)  # (batch_size ,k)
    multiply_pow = K.pow(K.sum(em, 1), 2)  # (batch_size, k)
    pair = 0.5 * (multiply_pow - pow_multiply)  # (batch_size, k)

    # deep
    deep = K.reshape(em, shape=(-1, int(feature_number * k)))
    for i in range(num_deep_layer):
        # deep  (batch_size, feature_number * k)
        deep = Dense(num_neuron,
                     use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1_deep, l2_deep),
                     name=f'deep_layer_{i}')(deep)
        deep = BatchNormalization()(deep) if deep_use_bn else deep
        deep_act = Activation(deep_activation) if isinstance(deep_activation, str) else deep_activation
        deep = deep_act(deep)
        deep = Dropout(rate=deep_dropout)(deep)  # (batch_size, layer)

    # concat & out
    out = linear   # (batch_size, 1)
    if use_fm:
        out = K.concatenate([out, pair], axis=-1)  # (batch_size, +=k)
    if use_deep:
        out = K.concatenate([out, deep], axis=-1)  # (batch_size, +=layer[-1])
    out = Dense(1, use_bias=True, activation=activation, name='out')(out)
    # model
    model = Model(inputs=[idx, val], outputs=out)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def xdeepfm(vocabulary_size, feature_number,
            activation, loss, metrics, optimizer,
            l1_linear=0., l2_linear=0.,
            l1_pair=0., l2_pair=0.,
            use_cin=True, num_cin_layer=2, num_cin_size=128,
            cin_activation='relu', split_half=True,
            l1_cin=0., l2_cin=0., cin_dropout=0.,
            l1_deep=0., l2_deep=0., use_deep=True,
            deep_dropout=0., deep_use_bn=False,
            num_deep_layer=2, num_neuron=128, deep_activation='relu', k=5):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # linear
    em_l = Embedding(vocabulary_size, 1,
                     input_length=feature_number,
                     name='em_l',
                     embeddings_regularizer=regularizers.L1L2(l1_linear, l2_linear))(idx)
    val_ed = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
    linear = K.sum(multiply([em_l, val_ed]), axis=1, keepdims=False)  # (batch_size, 1)

    # pair
    em_p = Embedding(vocabulary_size, k,
                     input_length=feature_number,
                     name='em_p',
                     embeddings_regularizer=regularizers.L1L2(l1_pair, l2_pair))(idx)  # (batch, feature_number, k)
    em = multiply([em_p, val_ed])    # (batch, feature_number, k)

    # deep
    deep = K.reshape(em, shape=(-1, int(feature_number * k)))
    for i in range(num_deep_layer):
        # deep  (batch_size, feature_number * k)
        deep = Dense(num_neuron,
                     use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1_deep, l2_deep),
                     name=f'deep_layer_{i}')(deep)
        deep = BatchNormalization()(deep) if deep_use_bn else deep
        deep_act = Activation(deep_activation) if isinstance(deep_activation, str) else deep_activation
        deep = deep_act(deep)
        deep = Dropout(rate=deep_dropout)(deep)  # (batch_size, layer)
    """
    # cin
    cin_list = [em]
    cin_reslut = []
    x0 = tf.split(cin_list[0], k * [1], 2)
    for i in range(num_cin_layer):

        fv = cin_list[-1].shape[1]  # feature vector
        fn = cin_list[0].shape[1]   # number of features
        xk = tf.split(cin_list[-1], k * [1], 2)
        cin = tf.matmul(x0, xk, transpose_b=True, name=f'cin_mat_{i}')  # (k, batch, num_feature, last feature vector)
        cin = tf.reshape(cin, shape=[k, -1, fn * fv])  # (k, batch, num_feature * last feature vector)
        cin = tf.transpose(cin, perm=[1, 0, 2])  # (batch, k, num_feature * last tensor feature vector)

        cin = Conv1D(filters=num_cin_size,
                     kernel_size=1,
                     padding='valid',
                     activation=cin_activation,
                     use_bias=True,
                     kernel_regularizer=regularizers.L1L2(l1_cin, l2_cin))(cin)   # (batch, k, feature_vector)
        cin = tf.transpose(cin, perm=[0, 2, 1])   # (batch, feature_vector, k)

        if split_half:
            if i != num_cin_layer - 1:
                next_hidden, direct_connect = tf.split(cin, 2 * [num_cin_size // 2], 1)
            else:
                direct_connect = cin
                next_hidden = 0
        else:
            direct_connect = cin
            next_hidden = cin

        cin_reslut.append(direct_connect)
        cin_list.append(next_hidden)
    """
    # cin
    cin_list = [tf.transpose(em, perm=[0, 2, 1])]
    cin_reslut = []
    x0 = K.expand_dims(cin_list[0])   # (batch, k, f0, 1)
    for i in range(num_cin_layer):

        fv = cin_list[-1].shape[2]  # feature vector
        fn = cin_list[0].shape[2]  # number of features

        xk = K.expand_dims(cin_list[-1])   # (batch, k, fk-1, 1)
        cin = tf.matmul(x0, xk, transpose_b=True, name=f'cin_mat_{i}')  # (batch, k, f0, fk-1)
        cin = tf.reshape(cin, shape=[-1, k, fn * fv])  # (batch, k, f0 * fk-1)

        cin = Conv1D(filters=num_cin_size,
                     kernel_size=1,
                     padding='valid',
                     activation=cin_activation,
                     use_bias=True,
                     kernel_regularizer=regularizers.L1L2(l1_cin, l2_cin))(cin)  # (batch, k, fv)

        if split_half:
            if i != num_cin_layer - 1:
                next_hidden, direct_connect = tf.split(cin, 2 * [num_cin_size // 2], 2)
            else:
                direct_connect = cin
                next_hidden = 0
        else:
            direct_connect = cin
            next_hidden = cin

        cin_reslut.append(direct_connect)
        cin_list.append(next_hidden)

    cin_reslut = tf.concat(cin_reslut, axis=2)   # (batch_size, k, feature map num)
    cin_reslut = tf.reduce_sum(cin_reslut, 1, keepdims=False)  # (batch_size, cin feature map num)
    cin_reslut = Dropout(rate=cin_dropout)(cin_reslut)  # (batch_size, layer)
    # concat & out
    out = linear   # (batch_size, 1)
    if use_cin:
        out = K.concatenate([out, cin_reslut], axis=-1)  # (batch_size, +=k)
    if use_deep:
        out = K.concatenate([out, deep], axis=-1)  # (batch_size, +=layer[-1])
    out = Dense(1, use_bias=True, activation=activation, name='out')(out)
    # model
    model = Model(inputs=[idx, val], outputs=out)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model
