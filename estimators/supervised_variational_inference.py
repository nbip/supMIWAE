import warnings

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils

from .variational_inference import VariationalInference, VariationalInferenceRegression


class SupervisedVariationalInference(VariationalInference):
    def __init__(self, **kwargs):
        super(SupervisedVariationalInference, self).__init__(**kwargs)

        self._trainable_parts = ['enc', 'dec', 'disc']

    def __call__(self, model, inputs, **kwargs):

        x, s, y = inputs
        # TODO: Instead of the below, maybe something like, if len(inputs) == 2: super().__call__()

        outputs = model((x, s), **kwargs)

        loss, metrics, samples = self.compute_supervised_elbo(inputs, outputs, **kwargs)

        outputs.update(samples)

        return loss, metrics, outputs

    def compute_supervised_elbo(self, inputs, outputs, **kwargs):

        x, s, y = inputs

        kl_weight = kwargs.get('kl_weight', 1)

        pyx, pxz, pz, qzx, z, x_samples, x_mixed = \
            [outputs[k] for k in ['pyx', 'pxz', 'pz', 'qzx', 'z', 'x_samples', 'x_mixed']]

        # ---- log probabilities
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        kl = -tf.reduce_mean(lpz - lqzx, axis=0)

        lpxz = tf.reduce_sum(pxz.log_prob(x) * s, axis=-1)

        # ---- log probabilities for the labels
        lpyx = pyx.log_prob(y)

        # ---- the regular iwae elbo
        iwae_log_w = lpxz + lpz - lqzx
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(iwae_log_w, axis=0), axis=-1)
        iwae_bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(iwae_log_w, axis=0) /
                                                   tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- self-normalized importance weights
        # ---- based only on covariates, not labels
        snis = tf.math.softmax(iwae_log_w, axis=0)

        # ---- log weights
        log_w = lpyx + lpxz + lpz - lqzx
        label_snis = tf.nn.softmax(log_w, axis=0)

        # ---- model elbo: logmeanexp over samples and average over batch
        elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
        bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(log_w, axis=0) /
                                              tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- importance weighted label probabilities -> predictions
        y_probs = tf.nn.softmax(pyx.logits, axis=-1)
        snis_y = tf.reduce_sum(snis[:, :, None] * y_probs, axis=0)
        preds = tf.cast(tf.math.argmax(snis_y, axis=-1), tf.float32)

        # ---- importance weighted x_samples
        snis_x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)
        snis_x_mixed = tf.reduce_sum(snis[:, :, None] * x_mixed, axis=0)

        # ---- importance weighted variational posterior samples
        snis_z = tf.reduce_sum(snis[:, :, None] * z, axis=0)

        # ---- monitor importance weighted lpyx
        snis_lpyx = tf.reduce_logsumexp(tf.math.log(snis) + lpyx, axis=0)

        # ---- monitor importance weighted lpxz
        snis_lpxz = tf.reduce_logsumexp(tf.math.log(snis) + lpxz, axis=0)

        # ---- accuracy
        hits = tf.cast(tf.equal(preds, y), tf.float32)
        acc = tf.reduce_mean(hits)

        metrics = {"loss": -elbo,
                   "elbo": elbo,
                   "bits_pr_observed_dim": bits_pr_observed_dim,
                   "iwae_elbo": iwae_elbo,
                   "iwae_bits_pr_observed_dim": iwae_bits_pr_observed_dim,
                   "lpz": lpz,
                   "lpxz": lpxz,
                   "lpyx": lpyx,
                   "lqzx": lqzx,
                   "kl": kl,
                   "snis_lpyx": snis_lpyx,
                   "snis_lpxz": snis_lpxz,
                   "acc": acc}

        samples = {"snis": snis,
                   "snis_x_samples": snis_x_samples,
                   "snis_x_mixed": snis_x_mixed,
                   "snis_z": snis_z,
                   "snis_y": snis_y,
                   "label_snis": label_snis,
                   "z": z,
                   "x_samples": x_samples,
                   "x_mixed": x_mixed,
                   "y_probs": y_probs,
                   "preds": preds,
                   "hits": hits}

        return -elbo, metrics, samples

    @tf.function
    def train_step(self, inputs, model, optimizer, **kwargs):

        with tf.GradientTape() as tape:
            loss, metrics, outputs = self(model, inputs, **kwargs)

        # ---- select which parts of the model to update
        map = {'enc': model.encoder.trainable_weights,
               'dec': model.decoder.trainable_weights,
               'disc': model.discriminator.trainable_weights}
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
              .format(epoch, epochs, metrics['elbo'].numpy(), val_metrics['elbo'].numpy(), took))
        print("\ttrain acc: {0:.4f}, val acc: {1:.4f}".format(metrics["acc"].numpy(), val_metrics["acc"].numpy()))

    @property
    def trainable_parts(self):
        return self._trainable_parts

    @trainable_parts.setter
    def trainable_parts(self, parts):
        self._trainable_parts = parts

    def single_imputation(self, inputs, model, **kwargs):
        """ Unsupervised/supervised single imputation.
        There are two versions of the self-normalized importance samples
        1) without p(y|x): 'snis'
        2) with p(y|x): 'label_snis'
        including p(y|x) in the importance weights leads to supervised imputations
        ----------
        :param inputs: list of tensors, (x, s, y) or (x, s). If y is provided, the imputations will be label guided
        :param model:
        :param kwargs: [n_samples: int, pareto: bool]
        :return: xhat, imputed data matrix
        """

        if len(inputs) == 3:
            # supervised imputation
            x, s, y = inputs
            snis_key = 'label_snis'
        else:
            # unsupervised imputation
            x, s = inputs
            snis_key = 'snis'

        xhat = np.nan * np.ones_like(x)

        pareto = kwargs.get('pareto', False)

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Single imputation', total=len(x))):

            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            # ---- feed through the model
            loss, metrics, outputs = self(model, input, **kwargs)

            # ---- pareto-smooth the importance samples
            if pareto:
                snis = outputs[snis_key].numpy()
                psis_lw, kss = utils.psislw(np.log(snis))
                psis = utils.softmax(psis_lw, axis=0)
                snis = psis
                if np.sum(kss > 0.7) > 0:
                    warnings.warn("kss larger than 0.7")
            else:
                snis = outputs[snis_key].numpy()

            # ---- importance weight the samples of missing data
            snis_x_mixed = np.sum(snis[:, :, None] * outputs['x_mixed'].numpy(), axis=0)

            xhat[i, :] = snis_x_mixed

        return xhat


class SupervisedVariationalInferenceRegression(VariationalInferenceRegression):
    def __init__(self, **kwargs):
        super(SupervisedVariationalInferenceRegression, self).__init__(**kwargs)

    def __call__(self, model, inputs, **kwargs):

        x, s, y = inputs
        # TODO: Instead of the below, maybe something like, if len(inputs) == 2: super().__call__()

        outputs = model((x, s), **kwargs)

        loss, metrics, samples = self.compute_supervised_elbo(inputs, outputs, **kwargs)

        outputs.update(samples)

        return loss, metrics, outputs

    def compute_supervised_elbo(self, inputs, outputs, **kwargs):

        x, s, y = inputs

        pyx, pxz, pz, qzx, z, x_samples, x_mixed = \
            [outputs[k] for k in ['pyx', 'pxz', 'pz', 'qzx', 'z', 'x_samples', 'x_mixed']]

        # ---- log probabilities
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        kl = -tf.reduce_mean(lpz - lqzx, axis=0)

        lpxz = tf.reduce_sum(pxz.log_prob(x) * s, axis=-1)

        # ---- log probabilities for the labels
        lpyx = tf.reduce_sum(pyx.log_prob(y), axis=-1)

        # ---- the regular iwae elbo
        iwae_log_w = lpxz + lpz - lqzx
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(iwae_log_w, axis=0), axis=-1)
        iwae_bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(iwae_log_w, axis=0) /
                                                   tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- self-normalized importance weights
        # ---- based only on covariates, not labels
        snis = tf.math.softmax(iwae_log_w, axis=0)

        # ---- log weights
        log_w = lpyx + lpxz + lpz - lqzx
        label_snis = tf.nn.softmax(log_w, axis=0)

        # ---- model elbo: logmeanexp over samples and average over batch
        elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)
        bits_pr_observed_dim = tf.reduce_mean(-utils.logmeanexp(log_w, axis=0) /
                                              tf.math.log(2.) / tf.reduce_sum(s, axis=-1), axis=-1)

        # ---- importance weighted variational posterior samples
        y_preds = tf.reduce_sum(snis[:, :, None] * pyx.loc, axis=0)

        # ---- importance weighted x_samples
        snis_x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)
        snis_x_mixed = tf.reduce_sum(snis[:, :, None] * x_mixed, axis=0)

        # ---- importance weighted variational posterior samples
        snis_z = tf.reduce_sum(snis[:, :, None] * z, axis=0)

        # ---- monitor importance weighted lpyx
        snis_lpyx = tf.reduce_logsumexp(tf.math.log(snis) + lpyx, axis=0)

        # ---- monitor importance weighted lpxz
        snis_lpxz = tf.reduce_logsumexp(tf.math.log(snis) + lpxz, axis=0)

        # ---- mse, rmse
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_preds), axis=-1))
        mse = tf.reduce_mean(tf.square(y - y_preds), axis=-1)

        mean_rmse = tf.reduce_mean(rmse, axis=-1)
        mean_mse = tf.reduce_mean(mse, axis=-1)

        metrics = {"loss": -elbo,
                   "elbo": elbo,
                   "bits_pr_observed_dim": bits_pr_observed_dim,
                   "iwae_elbo": iwae_elbo,
                   "iwae_bits_pr_observed_dim": iwae_bits_pr_observed_dim,
                   "lpz": lpz,
                   "lpxz": lpxz,
                   "lpyx": lpyx,
                   "lqzx": lqzx,
                   "kl": kl,
                   "snis_lpyx": snis_lpyx,
                   "snis_lpxz": snis_lpxz,
                   "mean_rmse": mean_rmse,
                   "mean_mse": mean_mse,
                   "neg_mean_mse": -mean_mse}

        samples = {"snis": snis,
                   "snis_x_samples": snis_x_samples,
                   "snis_x_mixed": snis_x_mixed,
                   "snis_z": snis_z,
                   "label_snis": label_snis,
                   "z": z,
                   "x_samples": x_samples,
                   "x_mixed": x_mixed,
                   "y_preds": y_preds,
                   "rmse": rmse,
                   "mse": mse}

        return -elbo, metrics, samples

    @tf.function
    def train_step(self, inputs, model, optimizer, **kwargs):

        with tf.GradientTape() as tape:
            loss, metrics, outputs = self(model, inputs, **kwargs)

        # ---- select which parts of the model to update
        map = {'enc': model.encoder.trainable_weights,
               'dec': model.decoder.trainable_weights,
               'disc': model.discriminator.trainable_weights}
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
              .format(epoch, epochs, metrics['elbo'].numpy(), val_metrics['elbo'].numpy(), took))
        print("\ttrain mse: {0:.4f}, val mse: {1:.4f}".format(metrics["mean_mse"].numpy(), val_metrics["mean_mse"].numpy()))

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
            snis_lpyx = np.nan * np.ones(n)

        for i, input_ in enumerate(tqdm(zip(x, s), desc='Predict', total=len(x))):

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
                lpyx = tf.reduce_mean(pyx.log_prob(y[i]), axis=-1)
                snis_lpyx[i] = tf.reduce_logsumexp(tf.math.log(snis) + lpyx, axis=0)

        if y is not None:
            return y_preds, snis_lpyx
        else:
            return y_preds

