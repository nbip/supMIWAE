import numpy as np
import tensorflow as tf

from .base import BaseSave


class LearnableImputation(tf.keras.Model):
    def __init__(self,
                 input_shape=2,
                 train_mean=None,
                 **kwargs):
        super(LearnableImputation, self).__init__()  # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        # self.add_weight vs tf.Variable(): https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
        initial_value = train_mean if train_mean is not None else np.zeros((1, input_shape))
        self.learnable_imputation = tf.Variable(name="learnable_imputation",
                                                initial_value=initial_value,
                                                dtype=tf.float32)

    def call(self, inputs):

        x, s = inputs

        return s * x + (1 - s) * self.learnable_imputation


class LearnableImputationModel(BaseSave):
    def __init__(self,
                 input_shape=2,
                 disc=None,
                 train_mean=None,
                 **kwargs):
        super(LearnableImputationModel, self).__init__()  # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        self.learnable_imputation_layer = LearnableImputation(input_shape, train_mean, **kwargs)
        self.discriminator = disc

        # ---- for saving separate model parts, see BaseSave
        self._savable_parts = {'li': self.learnable_imputation_layer,
                               'disc': self.discriminator}

    def call(self, inputs, **kwargs):

        return self.discriminator(self.learnable_imputation_layer((inputs)))
