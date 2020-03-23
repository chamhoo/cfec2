"""
auther: leechh
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Activation, Embedding, add, multiply, Dropout, BatchNormalization, Conv1D, Layer, Flatten, Add
from .keraslayers import outputs, EM, EmLinear, FMLayer, DNNLayer, AddBias, CIN, InteractingLayer, CrossLayer, AFM, SENET, Bilinear


def linear_regression(vocabulary_size, feature_number,
                      activation, loss, metrics, optimizer, l1=0., l2=0.):
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    x = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1, l2=l2)([idx, val])
    # add_bias
    x = AddBias(name='bias')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


def fm(vocabulary_size, feature_number, activation, loss, metrics, optimizer,
       l1_linear=0., l2_linear=0., l1_pair=0., l2_pair=0., k=10, use_linear=True):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # em
    em = EM(vocabulary_size, feature_number, name='em_l', k=k, l1=l1_pair, l2=l2_pair)([idx, val])
    x = FMLayer()(em)
    if use_linear:
        linear = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1_linear, l2=l2_linear)([idx, val])
        x = add([linear, x])
    x = AddBias(name='bias')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


def deepfm(vocabulary_size, feature_number,
           activation, loss, metrics, optimizer,
           l1_linear=0., l2_linear=0.,
           l1_pair=0., l2_pair=0., use_fm=True,
           l1_deep=0., l2_deep=0., use_deep=True,
           deep_dropout=0., deep_use_bn=False, deep_use_bias=False,
           num_deep_layer=2, num_neuron=128, deep_activation='relu', k=5):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    x = list()
    # linear
    x.append(EmLinear(vocabulary_size, feature_number, name='em', l1=l1_linear, l2=l2_linear)([idx, val]))
    # em
    em_w, em_val = EM(vocabulary_size, feature_number, name='em', k=k, l1=l1_pair, l2=l2_pair)([idx, val])
    em = multiply([em_w, em_val])   # (batch, fn, k)
    em = K.reshape(em, shape=(-1, int(feature_number * k)))   # (batch, fn * k)
    if use_deep:
        x.append(DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer,
                          l1=l1_deep, l2=l2_deep,
                          dropout=deep_dropout, activation=deep_activation,
                          bn=deep_use_bn, use_bias=deep_use_bias)(em))
    if use_fm:
        x.append(FMLayer(name='FM')([em_w, em_val]))

    # concat & out
    x = K.concatenate(x, axis=-1)
    x = Dense(1, use_bias=True, name='out')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


def xdeepfm(vocabulary_size, feature_number,
            activation, loss, metrics, optimizer,
            l1_linear=0., l2_linear=0.,
            l1_pair=0., l2_pair=0.,
            use_linear=True, use_cin=True,
            num_cin_layer=2, num_cin_size=128,
            cin_activation='linear', split_half=True,
            l1_cin=0., l2_cin=0., cin_dropout=0.,
            l1_deep=0., l2_deep=0., use_deep=True,
            deep_dropout=0., deep_use_bn=False,
            num_deep_layer=2, num_neuron=128, deep_activation='relu', k=5):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # x
    x = list()
    em = EM(vocabulary_size, feature_number, k=5, name='em', l1=l1_pair, l2=l2_pair)([idx, val])
    em = multiply(list(em))
    if use_deep:
        deep = K.reshape(em, shape=(-1, int(feature_number * k)))
        x.append(DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer,
                          l1=l1_deep, l2=l2_deep, dropout=deep_dropout,
                          activation=deep_activation, bn=deep_use_bn, use_bias=False)(deep))
    if use_cin:
        x.append(CIN(num_cin_layer=num_cin_layer,
                     num_cin_size=num_cin_size,
                     cin_activation=cin_activation,
                     split_half=split_half,
                     l1_cin=l1_cin, l2_cin=l2_cin, 
                     cin_dropout=cin_dropout)(em))
    if use_linear:
        x.append(EmLinear(vocabulary_size, feature_number, name='em_linear', l1=l1_linear, l2=l2_linear)([idx, val]))

    x = K.concatenate(x, axis=-1)  # (batch_size, +=layer[-1])
    x = Dense(1, use_bias=True, activation=activation, name='out')(x)
    # model
    model = Model(inputs=[idx, val], outputs=x)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def dcn(vocabulary_size, feature_number,
        activation, loss, metrics, optimizer,
        l1_linear=0., l2_linear=0.,
        l1_em=0., l2_em=0.,
        l1_cross=0., l2_cross=0.,
        num_cross=2,
        l1_deep=0., l2_deep=0.,
        deep_dropout=0., deep_use_bn=False,
        deep_use_bias=False,
        num_deep_layer=2, num_neuron=128,
        deep_activation='relu', k=5,
        use_linear=True, use_cross=True, use_deep=True):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # em
    em_w, em_val = EM(vocabulary_size, feature_number, name='em', k=k, l1=l1_em, l2=l2_em)([idx, val])
    em = multiply([em_w, em_val])   # (batch, fn, k)
    em = K.reshape(em, shape=(-1, int(feature_number * k)))   # (batch, fn * k)

    out = list()
    if use_linear:
        out.append(EmLinear(vocabulary_size, feature_number,
                            name='em_l',
                            l1=l1_linear, l2=l2_linear)([idx, val]))
    if use_deep:
        out.append(DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer,
                            l1=l1_deep, l2=l2_deep,
                            dropout=deep_dropout, activation=deep_activation,
                            bn=deep_use_bn, use_bias=deep_use_bias)(em))
    if use_cross:
        out.append(CrossLayer(layer_num=num_cross, l1=l1_cross, l2=l2_cross)(em))

    out = K.concatenate(out, axis=-1)
    out = Dense(1, use_bias=True, name='out')(out)
    return outputs([idx, val], out, activation, loss, metrics, optimizer)


def afm(vocabulary_size, feature_number,
        activation, loss, metrics, optimizer,
        l1_linear=0., l2_linear=0.,
        l1_em=0., l2_em=0.,
        afm_l1=0., afm_l2=0.,
        num_att=8, k=5,
        use_linear=True):

    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # afm
    em = EM(vocabulary_size, feature_number, name='EM', k=k, l1=l1_em, l2=l2_em)([idx, val])
    x = AFM(feature_number=feature_number, attention_size=num_att, afm_l1=afm_l1, afm_l2=afm_l2, k=k)(em)
    if use_linear:
        linear = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1_linear, l2=l2_linear)([idx, val])
        x = add([linear, x])
    x = AddBias(name='bias')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


def auto_int(vocabulary_size, feature_number,
             activation, loss, metrics, optimizer,
             l1_linear=0., l2_linear=0.,
             l1_em=0., l2_em=0.,
             attem_size=8, head_num=2,
             use_res=True, num_att_layer=3,
             att_dropout=0., att_act='relu',
             l1_deep=0., l2_deep=0.,
             deep_dropout=0., deep_use_bn=False,
             deep_use_bias=False,
             num_deep_layer=2, num_neuron=128,
             deep_activation='relu', k=5,
             use_linear=True, use_att=True, use_deep=True):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # em
    em_w, em_val = EM(vocabulary_size, feature_number, name='em', k=k, l1=l1_em, l2=l2_em)([idx, val])
    em = multiply([em_w, em_val])   # (batch, fn, k)

    out = list()
    if use_linear:
        out.append(EmLinear(vocabulary_size, feature_number,
                            name='em_l',
                            l1=l1_linear, l2=l2_linear)([idx, val]))
    if use_deep:
        deep = K.reshape(em, shape=(-1, int(feature_number * k)))  # (batch, fn * k)
        out.append(DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer,
                            l1=l1_deep, l2=l2_deep,
                            dropout=deep_dropout, activation=deep_activation,
                            bn=deep_use_bn, use_bias=deep_use_bias)(deep))
    if use_att:
        att = em
        for i in range(num_att_layer):
            att = InteractingLayer(attem_size=attem_size,
                                   head_num=head_num,
                                   use_res=use_res,
                                   dropout=att_dropout,
                                   activation=att_act,
                                   name=f'att_{i}')(att)
        out.append(tf.keras.layers.Flatten()(att))

    out = K.concatenate(out, axis=-1)
    out = Dense(1, use_bias=True, name='out')(out)
    return outputs([idx, val], out, activation, loss, metrics, optimizer)


def fibinet(vocabulary_size, feature_number,
            activation, loss, metrics, optimizer,
            l2_linear=0., l2_pair=0.,
            se_ratio=3, se_activation='relu',
            em_use_bi=True, selike_bi=True,
            bi_type='all',
            l2_deep=0., use_deep=True,
            deep_dropout=0., deep_use_bn=False, deep_use_bias=False,
            num_deep_layer=2, num_neuron=128, deep_activation='relu',
            k=5, seed=23333):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # em
    em_w, em_val = EM(vocabulary_size, feature_number, name='em', k=k, l2=l2_pair)([idx, val])
    em = multiply([em_w, em_val])  # (batch, fn, k)
    # se
    se = SENET(ratio=se_ratio, activation=se_activation, seed=seed)(em)    # (batch, fn, k)
    # bi
    if em_use_bi:
        em = Bilinear(_type=bi_type, seed=seed)(em)   # (batch, fn(fn-1)/2, k)
    if selike_bi:
        se = Bilinear(_type=bi_type, seed=seed)(se)   # (batch, fn(fn-1)/2, k)
    x = K.concatenate([em, se], axis=1)   # (batch, fn(fn-1) or 2*fn, k)
    x = Flatten()(x)
    # deep
    if use_deep:
        x = DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer, l2=l2_deep,
                     dropout=deep_dropout, activation=deep_activation,
                     bn=deep_use_bn, use_bias=deep_use_bias)(x)

    x = Dense(1, use_bias=False, activation=None)(x)
    x = Add([x, EmLinear(vocabulary_size, feature_number, name='em_l', l2=l2_linear)([idx, val])])
    x = AddBias()(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)
