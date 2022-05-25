import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils

from .base import Base


class MaximumLikelihood(Base):
    def __init__(self, **kwargs):
        super(MaximumLikelihood, self).__init__(**kwargs)

    def __call__(self, model, inputs, **kwargs):

        x, s, y = inputs

        outputs = model((x, s))

        loss, metrics = self.compute_loss(inputs, outputs, **kwargs)

        return loss, metrics, outputs

    def compute_loss(self, inputs, outputs, **kwargs):

        x, s, y = inputs

        pyx = outputs

        logits = pyx.logits
        probs = tf.nn.softmax(logits, axis=-1)
        preds = tf.cast(tf.math.argmax(probs, axis=-1), tf.float32)

        # average over batch
        lpyx = pyx.log_prob(y)
        loss = -tf.reduce_mean(lpyx, axis=-1)

        # ---- accuracy
        hits = tf.cast(tf.equal(preds, y), tf.float32)
        acc = tf.reduce_mean(hits)

        metrics = {"loss": loss,
                   "lpyx": lpyx,
                   "logits": logits,
                   "probs": probs,
                   "hits": hits,
                   "acc": acc}

        return loss, metrics

    @staticmethod
    def predict(inputs, model, **kwargs):
        x, s = inputs
        pyx = model((x, s))
        probs = utils.softmax(pyx.logits, axis=-1)
        preds = np.argmax(probs, axis=-1).astype(np.float32)

        return preds, probs

    @staticmethod
    def loop_predict(inputs, model, **kwargs):
        x, s = inputs
        n = x.shape[0]
        n_classes = model.discriminator.n_classes

        y_preds = np.nan * np.ones(n)
        y_probs = np.nan * np.ones((n, n_classes))

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Predict', total=len(x))):
            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            pyx = model(input)
            probs = utils.softmax(pyx.logits, axis=-1)
            preds = np.argmax(probs, axis=-1).astype(np.float32)

            y_preds[i] = preds
            y_probs[i, :] = probs

        return y_preds, y_probs

    @tf.function
    def train_step(self, inputs, model, optimizer, **kwargs):
        with tf.GradientTape() as tape:
            loss, metrics, outputs = self(model, inputs, **kwargs)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, metrics

    @tf.function
    def val_step(self, inputs, model, **kwargs):
        loss, metrics, outputs = self(model, inputs, **kwargs)
        return loss, metrics

    @staticmethod
    def print(metrics, val_metrics, epoch, epochs, took):
        print("epoch {0}/{1}, train loss: {2:.2f}, val loss: {3:.2f}, time: {4:.2f}"
              .format(epoch, epochs, metrics['loss'].numpy(), val_metrics['loss'].numpy(), took))
        print("\ttrain acc: {0:.4f}, val acc: {1:.4f}".format(metrics["acc"].numpy(), val_metrics["acc"].numpy()))


class MaximumLikelihoodRegression(MaximumLikelihood):
    def __init__(self, **kwargs):
        super(MaximumLikelihoodRegression, self).__init__(**kwargs)

    def compute_loss(self, inputs, outputs, **kwargs):

        x, s, y = inputs

        pyx = outputs

        y_preds = pyx.loc

        # log p(y|x)
        lpyx = tf.reduce_mean(pyx.log_prob(y), axis=-1)
        # average over batch
        loss = -tf.reduce_mean(lpyx, axis=-1)

        # ---- rmse, mse
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_preds), axis=-1))
        mse = tf.reduce_mean(tf.square(y - y_preds), axis=-1)

        mean_rmse = tf.reduce_mean(rmse, axis=-1)
        mean_mse = tf.reduce_mean(mse, axis=-1)

        metrics = {"loss": loss,
                   "lpyx": lpyx,
                   "y_preds": y_preds,
                   "rmse": rmse,
                   "mse": mse,
                   "mean_rmse": mean_rmse,
                   "mean_mse": mean_mse,
                   "neg_mean_mse": -mean_mse}

        return loss, metrics

    @staticmethod
    def predict(inputs, model, **kwargs):
        if len(inputs) == 2:
            x, s = inputs
            y = None
        else:
            x, s, y = inputs

        pyx = model((x, s))
        y_preds = pyx.loc

        if y is not None:
            lpyx = tf.reduce_mean(pyx.log_prob(y), axis=-1)
            return y_preds, lpyx
        else:
            return y_preds

    @staticmethod
    def loop_predict(inputs, model, **kwargs):
        x, s = inputs
        n = x.shape[0]
        n_out = model.discriminator.n_out

        y_preds = np.nan * np.ones((n, n_out))

        for i, input_ in enumerate(tqdm(zip(*inputs), desc='Predict', total=len(x))):
            # ---- add extra dimension to inputs, to mimick batch dimensions
            # ---- when looping over single elements
            input = [d[np.newaxis] for d in input_]

            pyx = model(input)
            y_preds[i] = pyx.loc

        return y_preds

    @staticmethod
    def print(metrics, val_metrics, epoch, epochs, took):
        print("epoch {0}/{1}, train loss: {2:.2f}, val loss: {3:.2f}, time: {4:.2f}"
              .format(epoch, epochs, metrics['loss'].numpy(), val_metrics['loss'].numpy(), took))
        print("\ttrain mse: {0:.4f}, val mse: {1:.4f}".format(metrics["mean_mse"].numpy(), val_metrics["mean_mse"].numpy()))
