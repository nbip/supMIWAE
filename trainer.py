import time

import utils
from utils import write_to_tensorboard


def train(model,
          estimator,
          optimizer,
          train_dataset,
          val_dataset,
          epochs,
          eval_every=100,
          performance_key='elbo',
          tensorboard=False,
          **kwargs):

    steps_pr_epoch = train_dataset.__len__()

    # ---- train
    start = time.time()
    best = -float("inf")

    for epoch in range(epochs):

        for _step, train_batch in enumerate(train_dataset):
            step = _step + steps_pr_epoch * epoch

            # ---- one training step
            loss, metrics = estimator.train_step(train_batch, model, optimizer, **kwargs)

            if step % eval_every == 0:

                took = time.time() - start
                start = time.time()

                # ---- write training to tensorboard
                if tensorboard:
                    with estimator.train_summary_writer.as_default():
                        write_to_tensorboard(metrics, step)

                # ---- monitor the val-set
                val_metrics = {}
                val_len = val_dataset.__len__().numpy()
                for val_batch in val_dataset:
                    val_loss, _val_metrics = estimator.val_step(val_batch, model, **kwargs)
                    utils.sum_metrics(val_metrics, _val_metrics)
                utils.scale_metrics(val_metrics, val_len)

                if tensorboard:
                    with estimator.test_summary_writer.as_default():
                        write_to_tensorboard(val_metrics, step)

                if val_metrics[performance_key] > best:
                    best = val_metrics[performance_key]
                    model.save_weights(filepath=estimator.save_dir, epoch=epoch, step=step)
                    print("Performance: {} is {:.4f}".format(performance_key, best))

                estimator.print(metrics, val_metrics, epoch, epochs, took)
