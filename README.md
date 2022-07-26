# supMIWAE
Code accompanying the paper [How to deal with missing data in supervised deep learning?](https://openreview.net/pdf?id=J7b4BCtDm4)

```bash
PYTHONPATH=. nohup python -u ./experiments/sklearn/task01.py --dataset half-moons --model supMIWAE --reps 5 --gpu "0" > log.log &
```
Tensorboard will be logged in `/tmp/sklearn/task01/`.
