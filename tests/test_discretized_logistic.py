import os
import unittest

import numpy as np
import tensorflow as tf

import utils


class TestDiscretizedLogistic(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(123)
        self.batch_size = 64
        self.img1 = 2 * tf.random.uniform([self.batch_size, 784]) - 1
        self.img2 = tf.constant(np.linspace(-1, 1, 256))

    def test_interval_width(self):
        dl = utils.DiscretizedLogistic(loc=0., logscale=0., low=0., high=255., levels=256.)

        self.assertEqual(dl.interval_width, 1.)

        dl = utils.DiscretizedLogistic(loc=0., logscale=0., low=-1., high=1., levels=256.)

        self.assertEqual(dl.interval_width, 2. / 255.)

    def test_edges(self):

        # distribution centered at the left edge
        dl = utils.DiscretizedLogistic(loc=0., logscale=1., low=0., high=1., levels=256.)

        # I guess the prob at 0 should be the cdf from -inf to 0 + interval_width/2
        # i.e. a little more than 0.5
        l = dl.log_prob(0.)
        print(tf.exp(l))

        # and we can do something similar on the right edge
        dl = utils.DiscretizedLogistic(loc=1., logscale=1., low=0., high=1., levels=256.)
        l = dl.log_prob(1.)
        print(tf.exp(l))

    def test_peaked(self):

        # let's make a very peaked distribution around a pixel value
        dl = utils.DiscretizedLogistic(loc=125., logscale=-30., low=0., high=255., levels=256.)

        # does the log probability at this pixel value contain most of the probability mass?
        l = dl.log_prob(125.)
        # print(tf.exp(l))
        self.assertEqual(tf.exp(l).numpy(), 1.)

        # can the method handle evaluting the log probability at a pixel with very
        # little of the probability mass?
        l = dl.log_prob(10.)
        # print(tf.exp(l))
        self.assertAlmostEqual(tf.exp(l).numpy(), 0.)

    def test_sampling(self):

        loc = 125.
        logscale = 2.
        n_samples = 5

        dl = utils.DiscretizedLogistic(loc=125., logscale=2., low=0., high=255., levels=256.)

        samples = dl.sample()
        # print(samples)
        self.assertEqual(samples.shape, ())

        samples = dl.sample(n_samples)
        # print(samples)
        self.assertEqual(samples.shape, (n_samples,))

        samples = dl.sample(1000)
        # print(tf.reduce_mean(samples), np.std(samples.numpy()), np.sqrt(np.exp(2)**2 * np.pi**2 / 3))
        self.assertAlmostEqual(samples.numpy().mean(), loc, 0)
        self.assertAlmostEqual(samples.numpy().std(), np.sqrt(np.exp(logscale)**2 * np.pi**2 / 3), 0)


if __name__ == '__main__':
    # PYTHONPATH=. python ./tests/test_discretized_logistic.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=' ', help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
