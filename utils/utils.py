import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis, keepdims=True)
    return tf.squeeze(tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis, keepdims=True)) + max, axis)


def logsoftmax(logits, axis):
    """numerically stable log(softmax(logits))"""
    max = tf.reduce_max(logits, axis=axis, keepdims=True)
    return logits - max - tf.math.log(tf.reduce_sum(tf.exp(logits - max), axis=axis, keepdims=True))


def bernoullisample(x):
    return np.random.binomial(1, x, size=x.shape).astype('float32')


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mask(x, m):
    return (np.random.uniform(0, 1, x.shape) > m).astype(np.float32)


def mask(xnan):
    return (~np.isnan(xnan)).astype(np.float32)


def get_xnan(x, s):
    xnan = np.copy(x)
    xnan[s == 0] = np.nan
    return xnan


def write_to_tensorboard(metrics, step):
    for key, value in metrics.items():
        tf.summary.scalar(key, value.numpy().mean(), step=step)


def get_tf_dataset(inputs, batch_size, **kwargs):
    if len(inputs) == 3:
        x, s, y = inputs
        n = x.shape[0]

        data = tf.data.Dataset.from_tensor_slices(x.astype(np.float32))
        labels = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))
        mask = tf.data.Dataset.from_tensor_slices(s.astype(np.float32))

        dataset = (tf.data.Dataset.zip((data, mask, labels))
                   .shuffle(n).batch(batch_size).prefetch(4))
        return dataset

    elif len(inputs) == 2:
        x, s = inputs
        n = x.shape[0]

        data = tf.data.Dataset.from_tensor_slices(x.astype(np.float32))
        mask = tf.data.Dataset.from_tensor_slices(s.astype(np.float32))

        dataset = (tf.data.Dataset.zip((data, mask))
                   .shuffle(n).batch(batch_size).prefetch(4))
        return dataset
    else:
        return None


def get_rugplot(data, ax, alpha=None, height=0.1, **kwargs):

    x, s, y = data

    idx_full = np.sum(s, axis=-1) == 2
    idx_x1 = s[:, 1] == 0
    idx_x2 = s[:, 0] == 0

    x_fully_observed = x[idx_full, :]
    x1_only= x[idx_x1, 0]
    x2_only = x[idx_x2, 1]

    # limit the number of rugs to 100 for plot-visibility
    sns.rugplot(x=x1_only[:100], hue=y[idx_x1][:100], ax=ax, height=height, **kwargs)
    sns.rugplot(y=x2_only[:100], hue=y[idx_x2][:100], ax=ax, height=height, **kwargs)
    sns.scatterplot(x=x_fully_observed[:, 0],
                    y=x_fully_observed[:, 1],
                    hue=y[idx_full],
                    ax=ax, alpha=alpha)
    ax.get_legend().remove()


def get_rugplot_kde(data, ax, alpha=None, height=0.1, **kwargs):

    x, s, y = data

    idx_full = np.sum(s, axis=-1) == 2
    idx_x1 = s[:, 1] == 0
    idx_x2 = s[:, 0] == 0

    x_fully_observed = x[idx_full, :]
    x1_only= x[idx_x1, 0]
    x2_only = x[idx_x2, 1]

    # limit the number of rugs to 100 for plot-visibility
    sns.rugplot(x=x1_only[:100], hue=y[idx_x1][:100], ax=ax, height=height)
    sns.rugplot(y=x2_only[:100], hue=y[idx_x2][:100], ax=ax, height=height)
    sns.kdeplot(x=x_fully_observed[:, 0],
                y=x_fully_observed[:, 1],
                hue=y[idx_full],
                ax=ax, alpha=alpha, **kwargs)
    ax.get_legend().remove()


def accuracy(y, ypred):
    return np.mean(y == ypred)


def rmse(y, ypred, axis=-1):
    return np.sqrt(np.mean((y - ypred)**2, axis=axis))


def mse(y, ypred, axis=-1):
    return np.mean((y - ypred)**2, axis=axis)


def mae(y, ypred, axis=-1):
    return np.mean(np.abs(y - ypred), axis=axis)


def get_accs(y, ypred, s):
    """Split acc into data affected by missing and not"""
    acc = accuracy(y, ypred)

    # ---- split by missing and observed
    idx = s.sum(-1) < 2
    acc_on_missing = accuracy(y[idx], ypred[idx])
    acc_on_observed = accuracy(y[~idx], ypred[~idx])

    return acc, acc_on_missing, acc_on_observed


def log_pyx(y, probs):

    probs = np.maximum(probs, np.finfo('float').tiny)
    pyx = tfd.Categorical(probs)

    return pyx.log_prob(y).numpy().mean()


def get_lpyx(y, probs, s):
    """Split lpyx into data affected by missing and not"""

    lpyx = log_pyx(y, probs)

    # ---- split by missing and observed
    idx = s.sum(-1) < 2
    lpyx_on_missing = log_pyx(y[idx], probs[idx])
    lpyx_on_observed = log_pyx(y[~idx], probs[~idx])

    return lpyx, lpyx_on_missing, lpyx_on_observed


def sum_metrics(metrics, update):
    """ Does not account for uneven batch sizes """
    for key in update.keys():
        metrics[key] = metrics.get(key, 0) + tf.reduce_mean(update[key])

def scale_metrics(metrics, factor):
    for key in metrics.keys():
        metrics[key] /= factor
