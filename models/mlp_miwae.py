import tensorflow as tf
from tensorflow_probability import distributions as tfd

from models.base import BaseSave


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 n_out,
                 n_hidden,
                 activation,
                 **kwargs):
        super(BasicBlock, self).__init__()

        self.n_out = n_out

        layers = [tf.keras.layers.Dense(nh, activation=activation) for nh in n_hidden]
        layers += [tf.keras.layers.Dense(2 * n_out, activation=None)]
        self.mlp = tf.keras.Sequential(layers)

    def call(self, x):

        out = self.mlp(x)
        loc, log_scale = tf.split(out, num_or_size_splits=2, axis=-1)

        return tfd.Normal(loc, tf.nn.softplus(log_scale) + 1e-6)


class BasicBlockT(tf.keras.Model):
    def __init__(self,
                 n_out,
                 n_hidden,
                 activation,
                 **kwargs):
        super(BasicBlockT, self).__init__()

        self.n_out = n_out

        layers = [tf.keras.layers.Dense(nh, activation=activation) for nh in n_hidden]
        layers += [tf.keras.layers.Dense(3 * n_out, activation=None)]
        self.mlp = tf.keras.Sequential(layers)

    def call(self, x):

        out = self.mlp(x)
        loc, log_scale, log_df = tf.split(out, num_or_size_splits=3, axis=-1)

        return tfd.StudentT(loc=loc,
                            scale=tf.nn.softplus(log_scale) + 1e-6,
                            df=3 + tf.nn.softplus(log_df))


class MLPMIWAE(BaseSave):
    def __init__(self,
                 input_shape=2,
                 n_latent=2,
                 n_hidden=(50, 50, 50),
                 enc=None,
                 dec=None,
                 disc=None,
                 activation=tf.nn.elu,
                 **kwargs):

        super(MLPMIWAE, self).__init__()

        self.encoder = BasicBlock(n_latent, n_hidden, activation, **kwargs) if enc is None else enc
        self.decoder = BasicBlock(input_shape, list(reversed(n_hidden)), activation, **kwargs) if dec is None else dec
        self.discriminator = disc

        # ---- for saving separate model parts, see BaseSave
        self._savable_parts = {'enc': self.encoder,
                               'dec': self.decoder,
                               'disc': self.discriminator}

    def pxz_samples(self, pxz):
        return pxz.sample()

    def call(self, inputs, **kwargs):

        x, s = inputs

        # ---- prior p(z)
        pz = tfd.Normal(0, 1)

        # ---- variational posterior q(z|x)
        # qzx = self.encoder(tf.concat([x, s], axis=-1))
        qzx = self.encoder(x)

        z = qzx.sample(kwargs['n_samples'])

        # ---- observation model p(x|z)
        pxz = self.decoder(z)

        # ---- samples from the observation model
        x_samples = self.pxz_samples(pxz)

        # ---- mix observed data with samples of the missing data
        x_mixed = s[None, :, :] * x[None, :, :] + (1 - s[None, :, :]) * x_samples

        # ---- discriminator p(y|x)
        pyx = self.discriminator(x_mixed)

        return {'pyx': pyx, 'pxz': pxz, 'pz': pz, 'qzx': qzx, 'z': z, 'x_mixed': x_mixed, 'x_samples': x_samples}
