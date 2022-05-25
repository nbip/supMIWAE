import os

import numpy as np
import tensorflow as tf


class BaseSave(tf.keras.Model):
    """
    Enable saving model subparts separately
    """

    def __init__(self, **kwargs) -> None:
        super(BaseSave, self).__init__(**kwargs)

    @property
    def savable_parts(self):
        return self._savable_parts

    @savable_parts.setter
    def savable_parts(self, parts):
        self._savable_parts = parts

    def save_weights(self, filepath, parts=None, epoch=None, step=None, **kwargs):

        os.makedirs(filepath, exist_ok=True)
        if step:
            np.savez(os.path.join(filepath, 'step.npz'), epoch=epoch, step=step)
            print(">> Saving model at epoch {0} step {1}... ".format(epoch, step), end='', flush=True)
        else:
            print(">> Saving model... ", end='', flush=True)

        for part in self._savable_parts if parts is None else parts:
            path = os.path.join(filepath, part)
            self._savable_parts[part].save_weights(filepath=path, **kwargs)
            print('part: {}, '.format(part), end='', flush=True)

    def load_weights(self, filepath, parts=None, **kwargs):

        assert os.path.exists(filepath) is True, "Path does not exists"
        if os.path.exists(os.path.join(filepath, 'step.npz')):
            arr = np.load(os.path.join(filepath, 'step.npz'))
            print(">> Loading model from epoch {0} step {1}... ".format(arr['epoch'], arr['step']), end='', flush=True)
        else:
            print(">> Loading model... ", end='', flush=True)

        for part in self._savable_parts if parts is None else parts:
            path = os.path.join(filepath, part)
            try:
                self._savable_parts[part].load_weights(filepath=path, **kwargs)
                print('part: {}, '.format(part), end='', flush=True)
            except Exception as e:
                print("Exception: {}".format(e))
                raise Warning("\nWeights for {} do not seem available".format(part))
