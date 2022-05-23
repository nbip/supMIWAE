import datetime
import os
from typing import Any

import tensorflow as tf


class Base(object):
    def __init__(self, **kwargs: Any) -> None:
        super(Base, self).__init__(**kwargs)

        self.train_summary_writer = None
        self.test_summary_writer = None

    def init_tensorboard(self, experiment: str) -> None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "/tmp/{0}/".format(experiment) + current_time + "/train"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_log_dir = "/tmp/{0}/".format(experiment) + current_time + "/test"
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        # ---- directory for saving trained models
        self.save_dir = os.path.join('saved_models/', experiment)
        os.makedirs(self.save_dir, exist_ok=True)

