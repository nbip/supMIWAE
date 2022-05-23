import unittest

import tensorflow as tf

from utils import logmeanexp


class TestLogmeanexp(unittest.TestCase):

    def setUp(self) -> None:
        self.n_samples = 5
        self.batch_size = 64
        tf.random.set_seed(123)

    def test_random(self):
        elbo = tf.random.normal([self.n_samples, self.batch_size])

        reference = tf.math.log(tf.reduce_mean(tf.exp(elbo), axis=0))
        result = logmeanexp(elbo, axis=0)

        print(reference)
        print(result)

        self.assertAlmostEqual(0, tf.reduce_sum(reference - result).numpy(), 5)

    def test_ones(self):
        elbo = tf.ones([self.n_samples, self.batch_size])

        reference = tf.math.log(tf.reduce_mean(tf.exp(elbo), axis=0))
        result = logmeanexp(elbo, axis=0)

        print(reference)
        print(result)

        self.assertAlmostEqual(0, tf.reduce_sum(reference - result).numpy(), 5)

    def test_zeros(self):
        elbo = tf.zeros([self.n_samples, self.batch_size])

        reference = tf.math.log(tf.reduce_mean(tf.exp(elbo), axis=0))
        result = logmeanexp(elbo, axis=0)

        print(reference)
        print(result)

        self.assertAlmostEqual(0, tf.reduce_sum(reference - result).numpy(), 5)

    def test_large_negative_numbers(self):
        elbo = - 100 * tf.ones([self.n_samples, self.batch_size])

        # gives -inf
        _reference = tf.math.log(tf.reduce_mean(tf.exp(elbo), axis=0))
        reference = - 100 * tf.ones(self.batch_size)

        # should give tensor of -100
        result = logmeanexp(elbo, axis=0)

        print(_reference)
        print(result)

        self.assertAlmostEqual(0, tf.reduce_sum(reference - result).numpy(), 5)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=' ', help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
