"""
auther: leechh

Template：
class LayerName(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return None

    def get_config(self):
        config = super().get_config()
        return config

    def compute_output_shape(self, input_shape):
        return None
"""
import numpy as np
from itertools import combinations
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


class FMLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        w, val = inputs
        pow_multiply = K.sum(multiply([K.pow(w, 2), K.pow(val, 2)]), 1)  # (batch_size ,k)
        multiply_pow = K.pow(K.sum(multiply([w, val]), 1), 2)  # (batch_size, k)
        pair = 0.5 * K.sum((multiply_pow - pow_multiply), axis=1, keepdims=True)  # (batch_size, 1)
        return pair

    def get_config(self):
        config = super().get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


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

    def compute_output_shape(self, input_shape):
        return [input_shape, self.num_neuron]

    def get_config(self):
        params = super().get_config()
        params['num_layer'] = self.num_layer
        params['num_neuron'] = self.num_neuron
        params['l1'] = self.l1
        params['l2'] = self.l2
        params['dropout'] = self.dropout
        params['activation'] = self.activation
        params['bn'] = self.bn
        params['use_bias'] = self.use_bias
        return params


class CIN(Layer):
    def __init__(self, num_cin_layer=2, num_cin_size=128,
                 cin_activation='relu', split_half=True,
                 l1_cin=0., l2_cin=0., cin_dropout=0.,
                 **kwargs):

        super().__init__(**kwargs)
        self.num_cin_layer = num_cin_layer
        self.num_cin_size = num_cin_size
        self.cin_activation = cin_activation
        self.split_half = split_half
        self.l1_cin = l1_cin
        self.l2_cin = l2_cin
        self.cin_dropout = cin_dropout

    def build(self, input_shape):
        self.conv_lst = list()
        for i in range(self.num_cin_layer):
            self.conv_lst.append(Conv1D(filters=self.num_cin_size,
                                        kernel_size=1,
                                        padding='valid',
                                        activation=self.cin_activation,
                                        use_bias=True,
                                        kernel_regularizer=regularizers.L1L2(self.l1_cin, self.l2_cin)))
        self.dropout = Dropout(rate=self.cin_dropout)

    def call(self, inputs, **kwargs):
        # inputs shape is (batch, feature_number, k)
        k = inputs.shape[-1]
        cin_list = [tf.transpose(inputs, perm=[0, 2, 1])]
        cin_reslut = []
        x0 = K.expand_dims(cin_list[0])  # (batch, k, f0, 1)
        for i in range(self.num_cin_layer):

            fv = cin_list[-1].shape[2]  # feature vector
            fn = cin_list[0].shape[2]  # number of features

            xk = K.expand_dims(cin_list[-1])  # (batch, k, fk-1, 1)
            cin = tf.matmul(x0, xk, transpose_b=True, name=f'cin_mat_{i}')  # (batch, k, f0, fk-1)
            cin = tf.reshape(cin, shape=[-1, k, fn * fv])  # (batch, k, f0 * fk-1)
            cin = self.conv_lst[i](cin)  # (batch, k, fv)

            if self.split_half:
                if i != self.num_cin_layer - 1:
                    next_hidden, direct_connect = tf.split(cin, 2 * [self.num_cin_size // 2], 2)
                else:
                    direct_connect = cin
                    next_hidden = 0
            else:
                direct_connect = cin
                next_hidden = cin

            cin_reslut.append(direct_connect)
            cin_list.append(next_hidden)

        cin_reslut = tf.concat(cin_reslut, axis=2)  # (batch_size, k, feature map num)
        cin_reslut = tf.reduce_sum(cin_reslut, 1, keepdims=False)  # (batch_size, cin feature map num)
        cin_reslut = self.dropout(cin_reslut)  # (batch_size, layer)
        return cin_reslut

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_cin_size

    def get_config(self):
        params = super().get_config()
        params['num_cin_size'] = self.num_cin_size
        params['num_cin_layer'] = self.num_cin_layer
        params['cin_activation'] = self.cin_activation
        params['split_half'] = self.split_half
        params['l1_cin'] = self.l1_cin
        params['l2_cin'] = self.l2_cin
        params['cin_dropout'] = self.cin_dropout
        return params


class CrossLayer(Layer):

    def __init__(self, layer_num=2, l1=0., l2=0., **kwargs):
        self.layer_num = layer_num
        self.l1 = l1
        self.l2 = l2
        super().__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        self.kernels = [self.add_weight(name=f'cross_kernel_{i}',
                                        shape=(dim, 1),
                                        regularizer=regularizers.L1L2(self.l1, self.l2),
                                        trainable=True) for i in range(self.layer_num)]

        self.bias = [self.add_weight(name=f'cross_bias_{i}',
                                     shape=(dim, 1),
                                     initializer=tf.keras.initializers.zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
            dot_ = tf.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self):
        config = super().get_config()

        config['layer_num'] = self.layer_num
        config['l2'] = self.l2
        config['l1'] = self.l1
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class InteractingLayer(Layer):
    def __init__(self, attem_size=8, head_num=2, use_res=True, dropout=0., activation='relu', seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.attem_size = attem_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.dropout_value = dropout
        self.activation_value = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])

        self.W_Query = self.add_weight(
            name='query',
            shape=[embedding_size, self.attem_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed)
        )
        self.W_key = self.add_weight(
            name='key',
            shape=[embedding_size, self.attem_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1)
        )
        self.W_Value = self.add_weight(
            name='value',
            shape=[embedding_size, self.attem_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2)
        )
        if self.use_res:
            self.W_Res = self.add_weight(
                name='res',
                shape=[embedding_size, self.attem_size * self.head_num],
                dtype=tf.float32,
                initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 3)
            )

        # Be sure to call this somewhere!
        self.act = Activation(self.activation_value) if isinstance(self.activation_value, str) else self.activation_value
        self.dropout = Dropout(self.dropout_value)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = K.dot(inputs, self.W_Query)   # (batch, fn, attem_size * nhead)
        keys = K.dot(inputs, self.W_key)     # ~
        values = K.dot(inputs, self.W_Value)   # ~

        # head_num None F D
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))   # (nhead, batch, fn, attem_size)
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))       # ~
        values = tf.stack(tf.split(values, self.head_num, axis=2))   # ~

        inner_product = tf.matmul(querys, keys, transpose_b=True)  # (nhead, batch, fn, fn)
        normalized_att_scores = tf.nn.softmax(inner_product)

        result = tf.matmul(normalized_att_scores, values)  # head_num None F D
        result = tf.concat(tf.split(result, self.head_num), axis=-1)
        result = tf.squeeze(result, axis=0)  # None F D*head_num

        if self.use_res:
            result += K.dot(inputs, self.W_Res)
        result = self.act(result)
        result = self.dropout(result)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.attem_size * self.head_num

    def get_config(self):
        config = super().get_config()
        config['attem_size'] = self.attem_size
        config['head_num'] = self.head_num
        config['use_res'] = self.use_res
        config['seed'] = self.seed
        return config


class AFM(Layer):
    def __init__(self, feature_number, attention_size=8, afm_l1=0., afm_l2=0., k=5, dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.attention_size = attention_size
        self.afm_l1 = afm_l1
        self.afm_l2 = afm_l2
        self.k = k
        self.feature_number = feature_number
        self.dropout = dropout

    def build(self, input_shape):
        self.attention_w = self.add_weight(name='attention_w',
                                           shape=[self.k, self.attention_size],
                                           dtype='float32',
                                           initializer=tf.keras.initializers.GlorotNormal(),
                                           regularizer=regularizers.L1L2(self.afm_l1, self.afm_l2),
                                           trainable=True)

        self.attention_b = self.add_weight(name='attention_b',
                                           shape=[self.attention_size],
                                           dtype='float32',
                                           initializer=tf.keras.initializers.GlorotNormal(),
                                           trainable=True)

        self.attention_h = self.add_weight(name='attention_h',
                                           shape=[self.attention_size],
                                           dtype='float32',
                                           initializer=tf.keras.initializers.GlorotNormal(),
                                           trainable=True)

        self.attention_p = self.add_weight(name='attention_p',
                                           shape=[self.k, 1],
                                           dtype='float32',
                                           initializer=tf.keras.initializers.GlorotNormal(),
                                           trainable=True)

        self.dropout_layer = Dropout(self.dropout)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        em_w, em_val = inputs
        em = multiply([em_w, em_val])   # (batch, fn, k)

        ew_product_list = []
        for i in range(self.feature_number):
            for j in range(i + 1, self.feature_number):
                ew_product_list.append(tf.multiply(em[:, i, :], em[:, j, :]))  # (batch, k)

        product = tf.stack(ew_product_list)  # (fn*(fn-1)/2), batch, k)
        product = tf.transpose(product, perm=[1, 0, 2], name='product')  # (batch, fn*(fn-1)/2), k)

        x = tf.reshape(product, shape=(-1, self.k))
        x = tf.matmul(x, self.attention_w)
        x = tf.add(x, self.attention_b)
        num_interactions = int(self.feature_number * (self.feature_number-1) / 2)
        x = tf.reshape(x, shape=[-1, num_interactions, self.attention_size])   # (batch, fn*(fn-1)/2), k)

        x = tf.multiply(tf.nn.relu(x), self.attention_h)
        x = tf.exp(tf.reduce_sum(x, axis=2, keepdims=True))  # (batch, fn*(fn-1)/2), 1)
        x_sum = tf.reduce_sum(x, axis=1, keepdims=True)   # (batch, 1, 1)
        out = tf.divide(x, x_sum, name='attention_out')     # (batch, fn*(fn-1)/2), 1)

        atx_product = tf.reduce_sum(tf.multiply(out, product), axis=1, name='afm')  # (batch, k)
        atx_product = self.dropout_layer(atx_product)
        return tf.matmul(atx_product, self.attention_p)   # (batch, 1)

    def get_config(self):
        config = super().get_config()

        config['attention_size'] = self.attention_size
        config['afm_l1'] = self.afm_l1
        config['afm_l2'] = self.afm_l2
        config['k'] = self.k
        config['feature_number'] = self.feature_number
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1


class SENET(Layer):
    def __init__(self, ratio=3, activation='relu', seed=None, **kwargs):
        self.ratio = ratio
        self.seed = int(seed)
        self.activation = activation
        super().__init__(seed=seed, **kwargs)

    def build(self, input_shape):
        # input_shape = (batch, num_field, k)
        _, num_field, _ = input_shape
        self.dense = list()

        self.dense.append(Dense(
            name='se_dense_0',
            units=num_field//self.ratio,
            activation=self.activation,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))

        self.dense.append(Dense(
            name='se_dense_1',
            units=num_field,
            activation=self.activation,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed+1)))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        origin = inputs
        # se
        se = K.mean(inputs, axis=-1, keepdims=False)
        se = self.dense[0](se)
        se = self.dense[1](se)
        # re_weight
        return multiply([K.expand_dims(se), origin])

    def get_config(self):
        config = super().get_config()
        config['ratio'] = self.ratio
        config['seed'] = self.seed
        config['activation'] = self.activation
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class Bilinear(Layer):
    def __init__(self, _type, seed=None, **kwargs):
        self._type = _type
        self.seed = seed
        super().__init__(seed=seed, **kwargs)

    def build(self, input_shape):
        _, nf, k = input_shape

        if self._type == 'all':
            self.w = self.add_weight(shape=(k, k),
                                     initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                                     name='bi_w')

        elif self._type == 'each':
            self.w = [self.add_weight(shape=(k, k),
                                      initializer=tf.keras.initializers.GlorotNormal(seed=self.seed+i),
                                      name=f'bi_w_{i}') for i in range(nf-1)]

        elif self._type == 'interaction':
            self.w = {f'{i}_{j}': self.add_weight(
                shape=(k, k),
                initializer=tf.keras.initializers.GlorotNormal(seed=self.seed+i),
                name=f'bi_w_{i}_{j}') for i, j in combinations(range(nf), 2)}
        else:
            raise ValueError(f'Not find {self._type}, pls select in [all, each, interaction]')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, nf, k = inputs.shape
        inputs = tf.split(inputs, nf, axis=1)

        if self.bilinear_type == "all":
            p = [multiply([K.dot(i, self.w), j]) for i, j in combinations(inputs, 2)]

        elif self.bilinear_type == "each":
            p = [multiply([K.dot(inputs[i], self.w[i]), inputs[j]]) for i, j in combinations(range(nf), 2)]

        elif self.bilinear_type == "interaction":
            p = [multiply([K.dot(inputs[i], self.w[f'{i}_{j}']), inputs[j]]) for i, j in combinations(range(nf), 2)]
        else:
            raise NotImplementedError
        return K.concatenate(p, axis=1)   # (batch, field_interaction, k)

    def get_config(self):
        config = super().get_config()
        config['_type'] = self._type
        config['seed'] = self.seed
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.sum(range(input_shape[1]))), input_shape[-1]