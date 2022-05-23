import warnings

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils

from .base import Base


class VariationalInference(Base):
    def __init__(self, **kwargs):
        super(VariationalInference, self).__init__(**kwargs)

        self._trainable_parts = ['enc', 'dec']

    def __call__(self, model, inputs, **kwargs):

        # ---- remove possible labels
        if len(inputs) == 3:
            inputs = inputs[:2]

        x, s = inputs

        outputs = model((x, s), **kwargs)

        loss, metrics, samples = self.compute_elbo(inputs, outputs, **kwargs)

        outputs.update(samples)

        return loss, metrics, outputs

    def compute_elbo(self, inputs, outputs, **kwargs):
        x, s = inputs

        kl_weight = kwargs.get('kl_weight', 1)

        pyx, pxz, pz, qzx, z, x_samples, x_mixed = \
            [outputs[k] for k in ['pyx', 'pxz', 'pz', 'qzx', 'z', 'x_samples', 'x_mixed']]

        # ---- log probabilities
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        kl = -tf.reduce_mean(lpz - lqzx, axis=0)

        lpxz = tf.reduce_sum(pxz.log_prob(x) * s, axis=-1)

        # ---- the regular iwae elbo
        log_w = lpxz + kl_weight * (lpz - lqzx)
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
        iwae_bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(log_w, axis=0) /
                                                   tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- self-normalized importance weights
        # ---- based only on covariates, not labels
        snis = tf.math.softmax(log_w, axis=0)
        # m = tf.reduce_max(log_w, axis=0, keepdims=True)
        # log_w_minus_max = log_w - m
        # w = tf.exp(log_w_minus_max)
        # snis = w / tf.reduce_sum(w, axis=0, keepdims=True)

        # ---- importance weighted label probabilities -> predictions
        # TODO: could also match snis and probs in the logit domain instead, if applying logsoftmax to log_w
        y_probs = tf.nn.softmax(pyx.logits, axis=-1)
        snis_y = tf.reduce_sum(snis[:, :, None] * y_probs, axis=0)
        preds = tf.cast(tf.math.argmax(snis_y, axis=-1), tf.float32)

        # ---- importance weighted x_samples
        snis_x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)
        snis_x_mixed = tf.reduce_sum(snis[:, :, None] * x_mixed, axis=0)

        # ---- importance weighted variational posterior samples
        snis_z = tf.reduce_sum(snis[:, :, None] * z, axis=0)

        # ---- monitor importance weighted lpxz
        snis_lpxz = tf.reduce_logsumexp(tf.math.log(snis) + lpxz, axis=0)

        metrics = {"loss": -iwae_elbo,
                   "iwae_elbo": iwae_elbo,
                   "iwae_bits_pr_observed_dim": iwae_bits_pr_observed_dim,
                   "lpz": lpz,
                   "lpxz": lpxz,
                   "lqzx": lqzx,
                   "kl": kl,
                   "snis_lpxz": snis_lpxz}

        samples = {"z": z,
                   "x_samples": x_samples,
                   "x_mixed": x_mixed,
                   "log_w": log_w,
                   "snis": snis,
                   "snis_z": snis_z,
                   "snis_x_samples": snis_x_samples,
                   "snis_x_mixed": snis_x_mixed,
                   "snis_y": snis_y,
                   "y_probs": y_probs,
                   "preds": preds}

        return -iwae_elbo, metrics, samples

    @tf.function
    def train_step(self, inputs, model, optimizer, **kwargs):

        with tf.GradientTape() as tape:
            loss, metrics, outputs = self(model, inputs, **kwargs)

        # ---- select which parts of the model to update
        map = {'enc': model.encoder.trainable_weights,
               'dec': model.decoder.trainable_weights}
        trainable_weights = [map[part] for part in self._trainable_parts]
        trainable_weights = [val for sublist in trainable_weights for val in sublist]

        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))

        return loss, metrics

    @tf.function
    def val_step(self, inputs, model, **kwargs):
        loss, metrics, outputs = self(model, inputs, **kwargs)
        return loss, metrics

    @staticmethod
    def print(metrics, val_metrics, epoch, epochs, took):
        print("epoch {0}/{1}, train ELBO: {2:.2f}, val ELBO: {3:.2f}, time: {4:.2f}"
              .format(epoch, epochs, metrics['iwae_elbo'].numpy(), val_metrics['iwae_elbo'].numpy(), took))

    @property
    def trainable_parts(self):
        return self._trainable_parts

    @trainable_parts.setter
    def trainable_parts(self, parts):
        self._trainable_parts = parts

    def get_llh(self, inputs, model, **kwargs):

        x, s = inputs
        n = x.shape[0]

        llh = np.nan * np.ones(n)

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='log-likelihood', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            loss, metrics, outputs = self(model, input, **kwargs)
            llh[i] = metrics['iwae_elbo']

        return llh

    def predict(self, inputs, model, **kwargs):
        x, s = inputs
        n = x.shape[0]
        # n_classes = model.discriminator.trainable_weights[-1].shape[0]
        n_classes = model.discriminator.n_classes
        pareto = kwargs.get('pareto', False)

        y_preds = np.nan * np.ones(n)
        y_probs = np.nan * np.ones((n, n_classes))

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Predict', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            outputs = model(input, **kwargs)
            loss, metrics, samples = self.compute_elbo(input, outputs, **kwargs)
            outputs.update(samples)

            pyx = outputs['pyx']

            # ---- pareto-smooth the importance samples
            if pareto:
                snis = outputs['snis'].numpy()
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = utils.softmax(psis_lw, axis=0)
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")
            else:
                snis = outputs['snis'].numpy()

            # TODO: match snis and probs in logit domain instead?
            probs = utils.softmax(pyx.logits, axis=-1)
            snis_probs = np.sum(snis[:, :, None] * probs, axis=0)
            preds = np.argmax(snis_probs, axis=-1).astype(np.float32)

            y_preds[i] = preds
            y_probs[i, :] = snis_probs

        return y_preds, y_probs

    def single_imputation(self, inputs, model, **kwargs):
        """unsupervised single imputation
        ----------
        :param inputs: list of tensors, (x, s).
        :param model:
        :param kwargs: [n_samples: int, pareto: bool]
        :return: xhat, imputed data matrix
        """

        x, s = inputs

        xhat = np.nan * np.ones_like(x)

        pareto = kwargs.get('pareto', False)

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Single imputation', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            outputs = model(input, **kwargs)
            loss, metrics, samples = self.compute_elbo(input, outputs, **kwargs)
            outputs.update(samples)

            # ---- pareto-smooth the importance samples
            if pareto:
                snis = outputs['snis'].numpy()
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = utils.softmax(psis_lw, axis=0)
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")
            else:
                snis = outputs['snis'].numpy()

            # ---- importance weight the samples of missing data
            snis_x_mixed = np.sum(snis[:, :, None] * outputs['x_mixed'].numpy(), axis=0)

            xhat[i, :] = snis_x_mixed

        return xhat

    def multiple_imputation(self, inputs, model, **kwargs):

        x, s  = inputs
        n_imputations = kwargs.get('n_imputations', 5)
        n_samples = kwargs.get('n_samples', 50)
        pareto = kwargs.get('pareto', False)
        replace = kwargs.get('replace', True)

        if x.ndim == 2:
            n, d = x.shape
        else:
            d = x.shape
            n = 1

        multiple_x = np.nan * np.ones((n_imputations, n, d))

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Multiple imputations', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            # loss, metrics, outputs = self(model, input, **kwargs)
            outputs = model(input, **kwargs)
            loss, metrics, samples = self.compute_elbo(input, outputs, **kwargs)
            outputs.update(samples)

            snis = outputs['snis'].numpy()
            x_samples = outputs['x_samples'].numpy().squeeze(1)

            # ---- pareto-smooth the importance samples
            if pareto:
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = tf.nn.softmax(psis_lw, axis=0).numpy()
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")

            # ---- sampling importance resampling according to the self-normalized importance weights
            # TODO: use tfd.Categorical for sampling imputation indeces here, based on log_snis instead
            idx = np.random.choice(np.arange(n_samples),
                                   size=n_imputations,
                                   replace=replace,
                                   p=snis.squeeze())

            multiple_x[:, i, :] = x_samples[idx, :]

        return multiple_x

    def multiple_latent(self, inputs, model, **kwargs):

        x, s = inputs
        n_imputations = kwargs.get('n_imputations', 100)
        n_samples = kwargs.get('n_samples', 10000)
        n_latent = kwargs.get('n_latent', 2)
        pareto = kwargs.get('pareto', False)

        if x.ndim == 2:
            n, d = x.shape
        else:
            d = x.shape
            n = 1

        multiple_z = np.nan * np.ones((n_imputations, n, n_latent))

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Multiple imputations', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            # loss, metrics, outputs = self(model, input, **kwargs)
            outputs = model(input, **kwargs)
            loss, metrics, samples = self.compute_elbo(input, outputs, **kwargs)
            outputs.update(samples)

            snis = outputs['snis'].numpy()
            z_samples = outputs['z'].numpy().squeeze(1)

            # ---- pareto-smooth the importance samples
            if pareto:
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = tf.nn.softmax(psis_lw, axis=0).numpy()
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")

            # ---- sampling importance resampling according to the self-normalized importance weights
            replace = True
            idx = np.random.choice(np.arange(n_samples),
                                   size=n_imputations,
                                   replace=replace,
                                   p=snis.squeeze())

            multiple_z[:, i, :] = z_samples[idx, :]

        return multiple_z


class VariationalInferenceRegression(VariationalInference):
    def __init__(self, **kwargs):
        super(VariationalInferenceRegression, self).__init__(**kwargs)

    def compute_elbo(self, inputs, outputs, **kwargs):
        x, s = inputs

        pyx, pxz, pz, qzx, z, x_samples, x_mixed = \
            [outputs[k] for k in ['pyx', 'pxz', 'pz', 'qzx', 'z', 'x_samples', 'x_mixed']]

        # ---- log probabilities
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        kl = -tf.reduce_mean(lpz - lqzx, axis=0)

        lpxz = tf.reduce_sum(pxz.log_prob(x) * s, axis=-1)

        # ---- the regular iwae elbo
        log_w = lpxz + lpz - lqzx
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
        iwae_bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(log_w, axis=0) /
                                                   tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- self-normalized importance weights
        # ---- based only on covariates, not labels
        snis = tf.math.softmax(log_w, axis=0)
        # m = tf.reduce_max(log_w, axis=0, keepdims=True)
        # log_w_minus_max = log_w - m
        # w = tf.exp(log_w_minus_max)
        # snis = w / tf.reduce_sum(w, axis=0, keepdims=True)

        # ---- importance weighted loc -> predictions
        y_preds = tf.reduce_sum(snis[:, :, None] * pyx.loc, axis=0)

        # ---- importance weighted x_samples
        snis_x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)
        snis_x_mixed = tf.reduce_sum(snis[:, :, None] * x_mixed, axis=0)

        # ---- importance weighted variational posterior samples
        snis_z = tf.reduce_sum(snis[:, :, None] * z, axis=0)

        # ---- monitor importance weighted lpxz
        snis_lpxz = tf.reduce_logsumexp(tf.math.log(snis) + lpxz, axis=0)

        metrics = {"loss": -iwae_elbo,
                   "iwae_elbo": iwae_elbo,
                   "iwae_bits_pr_observed_dim": iwae_bits_pr_observed_dim,
                   "lpz": lpz,
                   "lpxz": lpxz,
                   "lqzx": lqzx,
                   "kl": kl,
                   "snis_lpxz": snis_lpxz}

        samples = {"z": z,
                   "x_samples": x_samples,
                   "x_mixed": x_mixed,
                   "log_w": log_w,
                   "snis": snis,
                   "snis_z": snis_z,
                   "snis_x_samples": snis_x_samples,
                   "snis_x_mixed": snis_x_mixed,
                   "y_preds": y_preds}

        return -iwae_elbo, metrics, samples

    def predict(self, inputs, model, **kwargs):
        if len(inputs) == 2:
            x, s = inputs
            y = None
        else:
            x, s, y = inputs

        n_out = model.discriminator.n_out
        n = x.shape[0]

        pareto = kwargs.get('pareto', False)

        y_preds = np.nan * np.ones((n, n_out))
        if y is not None:
            lpyx = np.nan * np.ones(n)

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Predict', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            outputs = model(input, **kwargs)
            loss, metrics, samples = self.compute_elbo(input, outputs, **kwargs)
            outputs.update(samples)

            pyx = outputs['pyx']

            # ---- pareto-smooth the importance samples
            if pareto:
                snis = outputs['snis'].numpy()
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = utils.softmax(psis_lw, axis=0)
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")
            else:
                snis = outputs['snis'].numpy()

            snis_preds = np.sum(snis[:, :, None] * pyx.loc, axis=0)

            y_preds[i] = snis_preds

            if y is not None:
                lpyx[i] = tf.reduce_mean(pyx.log_prob(y), axis=-1)

        if y is not None:
            return y_preds, lpyx
        else:
            return y_preds
