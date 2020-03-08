"""
auther: leechh
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Activation, Embedding, add, multiply, Dropout, BatchNormalization, Conv1D, Layer


def outputs(inputs, x_out, activation, loss, metrics, optimizer):
    # activation
    act = Activation(activation) if isinstance(activation, str) else activation
    y = act(x_out)
    # model
    model = Model(inputs=inputs, outputs=y)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


class EM(Layer):
    def __init__(self, vocabulary_size, feature_number, k=5, l1=0., l2=0., **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.feature_number = feature_number
        self.k = k
        self.l1 = l1
        self.l2 = l2

    def build(self, input_shape):
        self.em = Embedding(self.vocabulary_size, self.k,
                            input_length=self.feature_number,
                            name=self.name,
                            embeddings_regularizer=regularizers.L1L2(self.l1, self.l2))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        idx, val = inputs
        em_w = self.em(idx)
        val = K.expand_dims(val, axis=-1)  # (batch_size, feature_numbers, 1)
        return em_w, val

    def get_config(self):
        params = super().get_config()
        params['vocabulary_size'] = self.vocabulary_size
        params['feature_number'] = self.feature_number
        params['k'] = self.k
        params['l1'] = self.l1
        params['l2'] = self.l2
        return params

    def compute_output_shape(self, input_shape):
        batch, fn = input_shape   # (batch, feature_number) * 2
        return [batch, fn, self.k], [batch, fn, 1]


class EmLinear(EM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 1

    def call(self, inputs, **kwargs):
        w, val = super().call(inputs)
        return K.sum(multiply([w, val]), axis=1, keepdims=False)  # (batch_size, 1)


class EmFM(EM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        w, val = super().call(inputs)
        pow_multiply = K.sum(multiply([K.pow(w, 2), K.pow(val, 2)]), 1)  # (batch_size ,k)
        multiply_pow = K.pow(K.sum(multiply([w, val]), 1), 2)  # (batch_size, k)
        pair = 0.5 * K.sum((multiply_pow - pow_multiply), axis=1, keepdims=True)  # (batch_size, 1)
        return pair


class AddBias(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w_bias',
                                  shape=[input_shape[-1]],
                                  dtype='float32',
                                  initializer='zeros',
                                  trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.w0

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        params = super().get_config()
        return params


class DNNLayer(Layer):
    def __init__(self, num_neuron=128, num_layer=2,
                 l1=0., l2=0., dropout=0.,
                 activation='relu', bn=False, use_bias=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.num_layer = num_layer
        self.num_neuron = num_neuron
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.activation = activation
        self.bn = bn
        self.use_bias = use_bias

    def build(self, input_shape):
        self.dense_lst = []
        self.bn_lst = []
        self.act_lst = []
        self.dropout_lst = []

        for i in range(self.num_layer):
            self.dense_lst.append(
                Dense(units=self.num_neuron,
                      use_bias=self.use_bias,
                      kernel_regularizer=regularizers.L1L2(self.l1, self.l2), name=f'dense_{i}')
            )

            self.bn_lst.append(BatchNormalization())
            act = Activation(self.activation) if isinstance(self.activation, str) else self.activation
            self.act_lst.append(act)
            self.dropout_lst.append(Dropout(rate=self.dropout))

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.num_layer):
            x = self.dense_lst[i](x)
            x = self.bn_lst[i](x)
            x = self.act_lst[i](x)
            x = self.dropout_lst[i](x)
        return x


def linear_regression(vocabulary_size, feature_number,
                      activation, loss, metrics, optimizer, l1=0., l2=0.):
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    x = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1, l2=l2)([idx, val])
    # add_bias
    x = AddBias(name='bias')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


def fm(vocabulary_size, feature_number,
       activation, loss, metrics, optimizer, l1_linear=0., l2_linear=0., l1_pair=0., l2_pair=0., k=10):
    # input
    idx, val = Input(shape=[feature_number], dtype='float32'), Input(shape=[feature_number], dtype='float32')
    # em
    linear = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1_linear, l2=l2_linear)([idx, val])
    pair = EmFM(vocabulary_size, feature_number, name='em_l', k=k, l1=l1_pair, l2=l2_pair)([idx, val])
    x = add([linear, pair])
    x = AddBias(name='bias')(x)
    return outputs([idx, val], x, activation, loss, metrics, optimizer)


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
    linear = EmLinear(vocabulary_size, feature_number, name='em_l', l1=l1_linear, l2=l2_linear)([idx, val])

    # deep
    em = EM(vocabulary_size, feature_number, k=5, name='em', l1=l1_pair, l2=l2_pair)([idx, val])
    em = multiply(em)
    deep = K.reshape(em, shape=(-1, int(feature_number * k)))
    deep = DNNLayer(num_neuron=num_neuron, num_layer=num_deep_layer,
                    l1=l1_deep, l2=l2_deep, dropout=deep_dropout,
                    activation=deep_activation, bn=deep_use_bn, use_bias=False)(deep)
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
