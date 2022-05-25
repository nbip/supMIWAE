# https://github.com/microsoft/EDDI
# https://github.com/steveli/partial-encoder-decoder/blob/master/time-series/toy_pvae.py
# /Users/nbip/proj/python/python-IWAE/iwae-18/DISC.py line 168
# /Users/nbip/proj/python/python-IWAE/iwae-18/tasks/uci/taskM6.py
import numpy as np
import tensorflow as tf

from .base import BaseSave


class PermutationInvariance(tf.keras.Model):
    def __init__(self,
                 input_shape=2,
                 embedding_size=20,
                 code_size=10,
                 activation=tf.nn.elu,
                 **kwargs):
        super(PermutationInvariance, self).__init__()

        # ---- embedding [1, input_shape, M]
        E_init = tf.keras.initializers.GlorotNormal()
        self.E = tf.Variable(name="embedding",
                             initial_value=E_init(shape=(1, input_shape, embedding_size)),
                             dtype=tf.float32)

        # ---- embedding bias [1, input_shape, 1]
        self.b = tf.Variable(name="embedding_bias",
                             initial_value=np.zeros((1, input_shape, 1)),
                             dtype=tf.float32)

        self.h = tf.keras.layers.Dense(code_size, activation=activation)

    def call(self, inputs):

        x, s = inputs

        # _batch_size = x.shape[0]
        _batch_size = tf.shape(x)[0]

        _E = tf.tile(self.E, [_batch_size, 1, 1])
        _b = tf.tile(self.b, [_batch_size, 1, 1])

        # ---- point net version: [E, x]
        x_aug = tf.concat([_E, _b, x[:, :, None]], axis=-1)

        # ---- zero-imputation generalization (point net plus): [E \times x]
        # x_aug = tf.concat([_E * x[:, :, None], _b, axis=-1)

        # ---- version from their code: [E \times x, x]
        # https://github.com/microsoft/EDDI/blob/master/p_vae.py#L91
        # x_aug = tf.concat([_E * x[:, :, None], _b, x[:, :, None]], axis=-1)

        # ---- map to code space
        _c = self.h(x_aug)

        # ---- zero out unobserved features
        _cz = _c * s[:, :, None]

        # ---- aggregation over feature dimension
        c = tf.reduce_sum(_cz, axis=1)

        return c


class PermutationInvarianceModel(BaseSave):
    def __init__(self,
                 input_shape=2,
                 embedding_size=20,
                 code_size=10,
                 disc=None,
                 activation=tf.nn.elu,
                 **kwargs):

        super(PermutationInvarianceModel, self).__init__()  # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        self.permutation_invariance_layer = PermutationInvariance(input_shape, embedding_size, code_size, activation)
        self.discriminator = disc

        # ---- for saving separate model parts, see BaseSave
        self._savable_parts = {'pi': self.permutation_invariance_layer,
                               'disc': self.discriminator}

    def call(self, inputs):

        c = self.permutation_invariance_layer(inputs)

        return self.discriminator(c)
