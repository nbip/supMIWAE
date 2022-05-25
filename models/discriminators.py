# conv architecture:
# - https://github.com/iffsid/disentangling-disentanglement/blob/public/src/models/vae_fashion_mnist.py#L65
# - https://github.com/sksq96/pytorch-vae/blob/master/vae.py
# - https://www.tensorflow.org/tutorials/generative/cvae
# - https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# - https://www.sciencedirect.com/science/article/pii/S1319157821000227
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .base import BaseSave


class SimpleDiscriminatorModel(BaseSave):
    """
    Wrapper around an MLP classifier or Conv classifier
    """
    def __init__(self,
                 disc=None,  # MLPClassifier or ConvClassifier
                 **kwargs):
        super(SimpleDiscriminatorModel, self).__init__()  # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        self.discriminator = disc

        # ---- for saving separate model parts, see BaseSave
        self._savable_parts = {'disc': self.discriminator}

    def call(self, inputs, **kwargs):
        x, s = inputs
        return self.discriminator(x)


class MLPClassifier(tf.keras.Model):
    def __init__(self,
                 n_classes=2,
                 n_hidden=(50, 50, 50),
                 activation=tf.nn.elu,
                 **kwargs):
        super(MLPClassifier, self).__init__()   # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        self.n_classes = n_classes

        layers = [tf.keras.layers.Dense(nh, activation=activation) for nh in n_hidden]
        layers += [tf.keras.layers.Dense(n_classes, activation=None)]
        self.mlp = tf.keras.Sequential(layers)

    def call(self, x):

        logits = self.mlp(x)

        return tfd.Categorical(logits=logits)


class ConvClassifier(tf.keras.Model):
    def __init__(self,
                 n_classes=10,
                 image_shape=(28, 28, 1),
                 filters=16,
                 activation=tf.nn.elu,
                 **kwargs):
        super(ConvClassifier, self).__init__()  # not passing kwargs to super here because we're using kwargs to ignore extra keywords. Maybe not exactly kosher

        self.n_classes = n_classes

        self.image_shape = image_shape
        self.org_shape = None
        self.n_classes = n_classes

        self.cnn = tf.keras.Sequential([
            # ---- 14, 14, filters
            tf.keras.layers.Conv2D(filters=filters, kernel_size=2, strides=(2, 2), padding='valid'),
            tf.keras.layers.Activation(activation),
            # ---- 7, 7, 2 * filters
            tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=2, strides=(2, 2), padding='valid'),
            tf.keras.layers.Activation(activation),
            # ---- 3, 3, 4 * filters
            tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=2, strides=(2, 2), padding='valid'),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=n_classes, activation=None)
        ])

    def call(self, x):

        # ---- merge sample and batch dimensions
        x = self.merge_samples_and_batch(x)

        logits = self.cnn(x)

        # ---- unmerge sample and batch dimensions
        logits = self.unmerge_samples_and_batch(logits)

        return tfd.Categorical(logits=logits)

    def merge_samples_and_batch(self, x):
        self.org_shape = x.shape
        return tf.reshape(x, [-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]])

    def unmerge_samples_and_batch(self, x):
        return tf.reshape(x, [*self.org_shape[:-1], self.n_classes])
