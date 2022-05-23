import tensorflow as tf
from tensorflow_probability import distributions as tfd


class DiscretizedLogistic:
    """
    Discretized version of the logistic distribution f(x; mu, s) = e^{-(x - mu) / s} / s(1 + e^{-(x-mu)/s})^2

    resources:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
    https://github.com/NVlabs/NVAE/blob/master/distributions.py
    https://github.com/openai/vdvae/blob/main/vae_helpers.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    https://github.com/JakobHavtorn/hvae-oodd/blob/main/oodd/layers/likelihoods.py#L536
    https://arxiv.org/pdf/1701.05517.pdf
    https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb
    http://bjlkeng.github.io/posts/pixelcnn/
    https://bjlkeng.github.io/posts/autoregressive-autoencoders/
    https://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
    https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-cifar10-importance-sampling.ipynb
    https://github.com/nbip/sM2/blob/main/utils/discretized_logistic.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py#L34
    """

    def __init__(self, loc, logscale, low=-1., high=1., levels=256.):
        self.loc = loc
        self.logscale = logscale
        self.low = low
        self.high = high
        self.levels = levels

        self.interval_width = (high - low) / (levels - 1.)

        self.dx = self.interval_width / 2.

    def logistic_cdf(self, x):
        a = (x - self.loc) / tf.exp(self.logscale)
        return tf.nn.sigmoid(a)

    def logistic_log_prob_approx(self, x):
        """
        log pdf value times interval width as an approximation to the area under the curve in that interval
        """
        a = (x - self.loc) / tf.exp(self.logscale)
        log_pdf_val = - a - self.logscale - 2 * tf.nn.softplus(-a)
        return log_pdf_val + tf.cast(tf.math.log(self.interval_width), tf.float32)

    def log_prob(self, x):

        start = x - self.dx
        stop = x + self.dx

        # ---- true probability based on the CDF
        prob = self.logistic_cdf(stop) - self.logistic_cdf(start)

        # ---- safeguard prob by taking the maximum of prob and 1e-12
        prob = tf.math.maximum(prob, 1e-12)

        # ---- edge cases
        # left edge if x=0. All the cdf from -inf to (0 + interval_width/2)
        # right edge if x=255. All the cdf from (255 - interval_width/2) to inf
        a = (x - self.loc + self.dx) / tf.exp(self.logscale)
        b = (x - self.loc - self.dx) / tf.exp(self.logscale)
        left_edge = a - tf.nn.softplus(a)
        right_edge = - tf.nn.softplus(b)

        # ---- approximated log prob, if the prob is too small
        log_prob_approx = self.logistic_log_prob_approx(x)

        # --- safeguard log_prob by taking the maximum of log_prob and log(1e-12)
        log_prob_approx = tf.math.maximum(log_prob_approx, tf.math.log(1e-12))

        # ---- use tf.where to choose between the true prob or the approixmation
        safe_log_prob = tf.where(prob > 1e-5, tf.math.log(prob), log_prob_approx)

        # ---- use tf.where to select the edge case probabilities when relevant
        safe_log_prob_with_edges = tf.where(tf.less_equal(x, self.low), left_edge, safe_log_prob)
        safe_log_prob_with_edges = tf.where(tf.greater_equal(x, self.high), right_edge, safe_log_prob_with_edges)

        return safe_log_prob_with_edges

    def sample(self, n_samples=[]):
        logistic_dist = tfd.Logistic(loc=self.loc, scale=tf.exp(self.logscale))
        samples = logistic_dist.sample(n_samples)
        samples = tf.clip_by_value(samples, self.low, self.high)

        return samples
