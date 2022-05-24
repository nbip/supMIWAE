"""
Experiments on SKLEARN datasets, only one feature with missing
"""
import argparse
import datetime
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.datasets import make_circles, make_moons
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting, enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trainer
import utils
from estimators import (
    MaximumLikelihood,
    SupervisedVariationalInference,
    VariationalInference,
)
from models import (
    MLPMIWAE,
    PPCA,
    LearnableImputationModel,
    MLPClassifier,
    PermutationInvarianceModel,
    SimpleDiscriminatorModel,
)

import seaborn as sns; sns.set()
import matplotlib; matplotlib.use('Agg')  # needed when running from commandline


EXPERIMENT = 'sklearn/task01'
RESULTS_DIR = os.path.join('results', EXPERIMENT)
ASSETS_DIR = os.path.join('assets', EXPERIMENT)


def induce_mcar_missing(x, m, random_state):
    """Select a feature, set elements to missing with probability m"""
    np.random.seed(random_state)
    xnan = np.copy(x)

    selected_feature = 1

    ix = (np.random.rand(len(xnan)) > m).astype(np.float32)
    xnan[ix == 0, selected_feature] = np.nan
    s = utils.mask(xnan)

    return xnan, s


def make_burger(n_samples, noise, random_state):
    # class 1 are the buns, class 0 is the beef
    # any constant imputation mechanism cannot handle this in terms of log p(y|x)
    # The Bayes error rate on fully observed data will be (cdf(1.) + 2 * cdf(1.) + cdf(1.)) / 3
    # The Bayes error rate on fully missing data will be 0.33
    np.random.seed(random_state)

    ns = int(n_samples / 3)

    x = np.concatenate([np.concatenate([np.random.uniform(-2, 2, ns)[:, None],
                                        2 + noise * np.random.randn(ns)[:, None]], axis=1),
                        np.concatenate([np.random.uniform(-2, 2, ns)[:, None],
                                        0 + noise * np.random.randn(ns)[:, None]], axis=1),
                        np.concatenate([np.random.uniform(-2, 2, ns)[:, None],
                                        -2 + noise * np.random.randn(ns)[:, None]], axis=1)], axis=0)

    y = np.concatenate([1 * np.ones(ns),
                        0 * np.ones(ns),
                        1 * np.ones(ns)])

    p = np.random.permutation(3 * ns)
    x = x[p, :]
    y = y[p]

    return x, y


def make_pinwheel(n_classes, samples_per_class, radial_std, tangential_std, rate, random_state=42):
    # code modified from Johnson et. al. (2016): https://github.com/mattjj/svae
    np.random.seed(random_state)

    rads = np.linspace(0, 2*np.pi, n_classes, endpoint=False)

    features = np.random.randn(n_classes*samples_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(n_classes), samples_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return data[:, 0:2], data[:, 2].astype(np.int)


def plot_data(data, save_str):
    rows, cols, size = 1, 1, 2.5

    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    utils.get_rugplot(data, ax, height=0.1)
    ax.set_aspect('equal')
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


def plot_imputed_data(data, save_str):
    x, s, y = data
    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, ax=ax)
    utils.get_rugplot_kde(data, ax, alpha=0.5, bw_adjust=0.5)
    sns.scatterplot(x=x[s[:, 1]==0, 0], y=x[s[:, 1]==0, 1], color='C3', ax=ax, edgecolor='k')
    ax.set_aspect('equal')
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


def plot_imputed_data_li(data, model, save_str):
    x, s, y = data
    li = model.learnable_imputation_layer.learnable_imputation.numpy().squeeze()

    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, ax=ax)
    utils.get_rugplot_kde(data, ax, alpha=0.5, bw_adjust=0.5)
    xm = x[s[:, 1]==0, 0]
    sns.scatterplot(x=xm, y=li[1] * np.ones(len(xm)), color='C3', ax=ax, edgecolor='k')
    ax.set_aspect('equal')
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


def plot_surface(data, model, save_str):

    X, S, y = data

    # ---- define a grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- evaluate the grid in the classifier
    if 'sklearn' in str(model.__class__):
        grid_probs = model.predict_proba(grid)[:, 1]
    elif args.model == 'permutation-invariance':
        pyx = model((grid, np.ones_like(grid).astype(np.float32)))
        grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]
    else:
        clf = model.discriminator
        pyx = clf(grid)
        grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]

    grid_probs = grid_probs.reshape(xx.shape)

    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.contourf(xx, yy, grid_probs, alpha=.5)
    fig.add_subplot(ax)
    plt.savefig(save_str + '_plain')
    plt.close()

    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.contourf(xx, yy, grid_probs, alpha=.5)
    utils.get_rugplot((X, S, y), ax)
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


def plot_single(data, model, save_str):

    x, s, y = data

    # ---- define a grid
    h = .02  # step size in the mesh
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- get grid probs
    pyx = model.discriminator(grid)
    grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]
    grid_probs = grid_probs.reshape(xx.shape)

    # ---- define a sequence of input points to get class probabilities from
    n = 100
    x = np.concatenate([np.linspace(x_min, x_max, n)[:, None], np.zeros(n)[:, None]], axis=1)
    s = np.concatenate([np.ones((n, 1)), np.zeros((n, 1))], axis=1).astype(np.float32)
    y = np.zeros(n)

    # ---- look at imputations
    estimator = VariationalInference()
    x_mixed = estimator.single_imputation((x, s), model, n_samples=10000, pareto=False)

    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    sns.scatterplot(x=x_mixed[:, 0], y=x_mixed[:, 1], color='C3', ax=ax, edgecolor='k')
    fig.add_subplot(ax)
    plt.savefig(save_str + '_plain')
    plt.close()

    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    utils.get_rugplot_kde(data, ax, alpha=0.5, bw_adjust=0.5)
    sns.scatterplot(x=x_mixed[:, 0], y=x_mixed[:, 1], color='C3', ax=ax, edgecolor='k')
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()



def plot_multiple(data, model, save_str):

    x, s, y = data

    # ---- define a grid
    h = .02  # step size in the mesh
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- get grid probs
    pyx = model.discriminator(grid)
    grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]
    grid_probs = grid_probs.reshape(xx.shape)

    # ---- look at multiple imputations across a range of xobs values
    xobs = np.asarray([[-1.5, 0.], [0., 0.], [1.5, 0.]]).astype(np.float32)
    sobs = np.ones_like(xobs)
    sobs[:, 1] = 0

    # ---- with SIR
    estimator = VariationalInference()
    multiple_x = estimator.multiple_imputation((xobs, sobs), model, n_imputations=500, n_samples=20000)
    x_mixed = sobs[None, :, :] * xobs[None, :, :] + (1 - sobs[None, :, :]) * multiple_x

    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    utils.get_rugplot_kde(data, ax, alpha=0.5, bw_adjust=0.5)
    for i in range(3):
        ax.scatter(x_mixed[:, i, 0], x_mixed[:, i, 1], color='C{}'.format(i + 2), edgecolor='k')
    # get the axes limits for the next plot
    ylim = ax.get_ylim()
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()

    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    for i in range(3):
        sns.kdeplot(y=x_mixed[:, i, 1], color='C{}'.format(i + 2), ax=ax, bw_adjust=0.5)
    # set the axes limits according to the previous plot
    ax.set_ylim(ylim)
    fig.add_subplot(ax)
    plt.savefig(save_str + '_kde')
    plt.close()


def plot_classprobs(data, model, save_str):
    # ---- for [x1, nan], show the class probabilities in two cases:
    # ---- 1) using single imputation
    # ---- 2) using our model

    X, S, y = data

    # ---- define a grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- get grid probs
    pyx = model.discriminator(grid)
    grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]
    grid_probs = grid_probs.reshape(xx.shape)

    # ---- define a sequence of input points to get class probabilities from
    xseq = np.linspace(x_min, x_max, 200)
    xseq = np.asarray([xseq, np.zeros_like(xseq)]).T

    sseq = np.ones_like(xseq).astype(np.float32)
    sseq[:, 1] = 0

    # ---- single imputation
    estimator = VariationalInference()
    imp = estimator.single_imputation((xseq, sseq), model, n_samples=10000, pareto=False)

    # ---- predict based on imputations
    _, impute_probs = estimator.predict((imp, np.ones_like(imp).astype(np.float32)), model, n_samples=10000, pareto=False)
    impute_probs = impute_probs[:, 1]

    # ---- predict directly using full model, no imputations
    _, probs = estimator.predict((xseq, sseq), model, n_samples=10000, pareto=False)
    probs = probs[:, 1]

    rows, cols, size = 2, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.contourf(xx, yy, grid_probs, alpha=.5)
    utils.get_rugplot((X, S, y), ax, alpha=0.5)
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, inner_grid[1])
    # ax.plot(xseq[:, 0], impute_probs, c='C3', label=r'single imptuation $p(y=1|x)$')
    ax.plot(xseq[:, 0], probs, c='C4', label=r"$p(y=1|x)$")
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # ax.set_axis_off()
    ax.legend()
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


    fig, ax = plt.subplots(2, 1, sharex='all')
    fig.set_figheight(2 * fig.get_figheight())
    ax[0].contourf(xx, yy, grid_probs, alpha=.5)
    utils.get_rugplot((X, S, y), ax[0], alpha=0.5)
    # utils.get_rugplot_kde((X, S, y), ax[0], alpha=0.5)
    ax[0].scatter(imp[:, 0], imp[:, 1], c='C3', alpha=0.5, label='single imputation')
    ax[0].legend()
    ax[1].plot(xseq[:, 0], impute_probs, c='C3', label='single imputation')
    ax[1].plot(xseq[:, 0], probs, c='C4', label=args.model)
    ax[1].legend()
    ax[1].set_ylabel(r'$p(y=1 | x)$')
    plt.savefig(save_str + '_org')
    plt.close()


def plot_posterior(data, model, save_str):

    x, s, y = data

    estimator = VariationalInference()

    loss, metrics, outputs = estimator(model, data, n_samples=100)
    snis_z = outputs['snis_z'].numpy()
    snis = outputs['snis'].numpy()
    n_latent = snis_z.shape[1]

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title("Latent distribution")
    sns.scatterplot(x=snis_z[:, 0], y=snis_z[:, 1], hue=y, ax=ax)
    ax.set_aspect('equal')
    plt.savefig(save_str + '_latent')
    plt.close()

    # ---- look at psis diagnostic
    psis_lw, kss = utils.psislw(np.log(snis.T))  # OBS! takes [n_samples, batch] shaped input!
    psis = tf.nn.softmax(psis_lw, axis=0).numpy().T

    fig, ax = plt.subplots(1, 2, sharey='all')
    ax[0].scatter(np.linspace(0, len(kss), len(kss)), kss)
    sns.violinplot(y=kss, orient="v", ax=ax[1])
    plt.tight_layout()
    plt.savefig(save_str + '_kss')
    plt.close()

    # ---- now inspect the full posterior of a single observation with missing values
    xobs = np.asarray([[-1.5, 0.], [0., 0.], [1.5, 0.]]).astype(np.float32)
    sobs = np.ones_like(xobs)
    sobs[:, 1] = 0

    multiple_z = estimator.multiple_latent((xobs, sobs), model,  n_latent=n_latent, n_samples=10000, n_imputations=100)

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title("Latent distribution")
    sns.scatterplot(x=snis_z[:, 0], y=snis_z[:, 1], hue=y, alpha=0.5, ax=ax)
    # sns.scatterplot(x=multiple_z[:, 0, 0], y=multiple_z[:, 0, 1], color='C4', ax=ax)
    for i in range(3):
        try:
            sns.kdeplot(x=multiple_z[:, i, 0], y=multiple_z[:, i, 1], color='C{}'.format(i + 2), ax=ax)
        except:
            print("Somethings fishy with this plot_posterior... ")
    ax.set_aspect('equal')
    plt.savefig(save_str + '_latent_dist_missing')
    plt.close()

    # ---- now inspect the full posterior of a single observation without missing values
    xobs = np.asarray([[-1.5, 0.], [0., 0.], [1.5, 0.]]).astype(np.float32)
    sobs = np.ones_like(xobs)

    multiple_z = estimator.multiple_latent((xobs, sobs), model, n_latent=n_latent, n_samples=10000, n_imputations=100)

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title("Latent distribution")
    sns.scatterplot(x=snis_z[:, 0], y=snis_z[:, 1], hue=y, alpha=0.5, ax=ax)
    # sns.scatterplot(x=multiple_z[:, 0, 0], y=multiple_z[:, 0, 1], color='C4', ax=ax)
    for i in range(3):
        try:
            sns.kdeplot(x=multiple_z[:, i, 0], y=multiple_z[:, i, 1], color='C{}'.format(i + 2), ax=ax)
        except:
            print("Somethings fishy with this plot_posterior... ")
    ax.set_aspect('equal')
    plt.savefig(save_str + '_latent_dist_observed')
    plt.close()


def plot_predictive(model, n_latent, save_str):
    from tensorflow_probability import distributions as tfd

    # ---- predictive plot: sample from the generative model
    zs = tfd.Normal(0, 1).sample([1000, n_latent])  # [batch, dim]
    pxz = model.decoder(zs)
    xs = pxz.sample()
    pyx = model.discriminator(xs)
    ys = pyx.sample()
    xs = xs.numpy()
    ys = ys.numpy()

    # ---- define a grid
    h = .02  # step size in the mesh
    x_min, x_max = xs[:, 0].min() - .5, xs[:, 0].max() + .5
    y_min, y_max = xs[:, 1].min() - .5, xs[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- get grid probs
    pyx = model.discriminator(grid)
    grid_probs = utils.softmax(pyx.logits, axis=-1)[:, 1]
    grid_probs = grid_probs.reshape(xx.shape)

    rows, cols, size = 1, 1, 2.5
    fig = plt.figure(figsize=(cols * size, rows * size))
    inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                   left=0, right=1, top=1, bottom=0)
    ax = plt.Subplot(fig, inner_grid[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.contourf(xx, yy, grid_probs, alpha=.5)
    sns.scatterplot(x=xs[:, 0], y=xs[:, 1], hue=ys, ax=ax, alpha=.5)
    ax.set_aspect('equal')
    fig.add_subplot(ax)
    plt.savefig(save_str)
    plt.close()


def main(args):

    # ---- load config
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # ---- logger
    results_file = os.path.join(RESULTS_DIR, args.dataset + '_' + args.model + '.pkl')
    results = pd.DataFrame(columns=['model', 'm'])

    # for random_state_m, m in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
    for random_state_m, m in enumerate([0.5]):
        m_str = str(m).replace('.', '')
        print("M: ", m)
        prefix = 'task01_{}_{}_{}'.format(args.dataset, args.model, m_str)

        for rep in range(args.reps):
            print("Rep: {0}/{1}".format(rep + 1, args.reps))
            random_state = random_state_m * args.reps + rep

            # ---- generate data
            data = dataset_map[args.dataset](**config[args.dataset]['data_settings'], random_state=random_state)

            Xtrain_, Xtest, ytrain_, ytest = train_test_split(*data, test_size=1000, random_state=random_state)
            Xtrain, Xval, ytrain, yval = train_test_split(Xtrain_, ytrain_, test_size=1000, random_state=random_state)

            # ---- add missing
            Xtrain_nan, Strain = induce_mcar_missing(Xtrain, m, random_state=random_state)
            Xval_nan, Sval = induce_mcar_missing(Xval, m, random_state=random_state)
            Xtest_nan, Stest = induce_mcar_missing(Xtest, m, random_state=random_state)

            # ---- standardize according to train set
            scaler = StandardScaler()
            Xtrain_nan = scaler.fit_transform(Xtrain_nan)
            Xval_nan = scaler.transform(Xval_nan)
            Xtest_nan = scaler.transform(Xtest_nan)

            # ---- impute?
            imputer = imputer_map[args.model]

            imputer.fit(Xtrain_nan)
            Xtrain_i = imputer.transform(Xtrain_nan)
            Xval_i = imputer.transform(Xval_nan)
            Xtest_i = imputer.transform(Xtest_nan)

            # ---- model ---- #
            if args.model != 'gb':
                clf = MLPClassifier(**config['mlp']['model_settings'],
                                    **config[args.dataset]['model_settings'])
                model = model_map[args.model](**config[args.model]['model_settings'], disc=clf)
            else:
                model = model_map[args.model](**config[args.model]['model_settings'])

            # ---- for M1-M3, pretrain a MIWAE
            if args.model in ['m1', 'm2', 'm3']:

                # ---- create tf datasets without labels for pretraining
                train_dataset = utils.get_tf_dataset((Xtrain_i, Strain), **config['training'])
                val_dataset = utils.get_tf_dataset((Xval_i, Sval), batch_size=Xval_i.shape[0])

                # ---- estimator for pretraining ---- #
                tf.random.set_seed(random_state)
                pretrainer = VariationalInference()
                pretrainer.init_tensorboard(os.path.join(EXPERIMENT, args.model))

                # ---- pretrain the generative part of the model
                optimizer = tf.keras.optimizers.Adam(1e-3)
                trainer.train(model,
                              pretrainer,
                              optimizer,
                              train_dataset,
                              val_dataset,
                              **config['pretrain'])

                # ---- load the best set of weights
                model.load_weights(pretrainer.save_dir)

            # ---- M1) train using the full ELBO.
            # ---- M2) train using the full ELBO, with fixed generator
            if args.model in ['m1', 'm2']:

                # ---- create tf datasets with labels
                train_dataset = utils.get_tf_dataset((Xtrain_i, Strain, ytrain), **config['training'])
                val_dataset = utils.get_tf_dataset((Xval_i, Sval, yval), batch_size=Xval_i.shape[0])

                # ---- initialize a new estimator, using the full ELBO
                estimator = SupervisedVariationalInference()
                estimator.init_tensorboard(os.path.join(EXPERIMENT, args.model))

                # in M2, fix the generator, only the classifier can be
                # trained, see the model.train_step
                trainable_parts = ['disc'] if args.model == 'm2' else ['enc', 'dec', 'disc']
                estimator.trainable_parts = trainable_parts

                optimizer = tf.keras.optimizers.Adam(1e-4)
                trainer.train(model,
                              estimator,
                              optimizer,
                              train_dataset,
                              val_dataset,
                              **config['training'])

                # ---- load best set of weights ( use accuracy or p(y|x) ? )
                model.load_weights(estimator.save_dir)

            # ---- M3) impute using the generative part of the model and train the classifier
            if args.model == 'm3':

                # ---- impute
                Xtrain_ii = pretrainer.single_imputation((Xtrain_i, Strain), model, n_samples=10000, pareto=False)
                Xval_ii = pretrainer.single_imputation((Xval_i, Sval), model, n_samples=10000, pareto=False)
                Xtest_ii = pretrainer.single_imputation((Xtest_i, Stest), model, n_samples=10000, pareto=False)

                # ---- pass the imputed datasets on to the classifier
                Xtrain_i = Xtrain_ii
                Xval_i = Xval_ii
                Xtest_i = Xtest_ii

                # ---- create imputed tf datasets
                # OBS! here the mask is fully observed! not that it matters for the classifier...
                train_dataset = utils.get_tf_dataset((Xtrain_i, np.ones_like(Xtrain_i).astype(np.float32), ytrain), **config['training'])
                val_dataset = utils.get_tf_dataset((Xval_i, np.ones_like(Xval_i).astype(np.float32), yval), batch_size=Xval_i.shape[0])

                # ---- train the classifier part of the model
                estimator = MaximumLikelihood()
                estimator.init_tensorboard(os.path.join(EXPERIMENT, args.model))

                m3_model = SimpleDiscriminatorModel(disc=clf)

                optimizer = tf.keras.optimizers.Adam(1e-4)
                trainer.train(m3_model,
                              estimator,
                              optimizer,
                              train_dataset,
                              val_dataset,
                              **config['training'])

                m3_model.load_weights(estimator.save_dir)

                # OBS! we are appending the clf classifier to the pretrained MIWAE
                # to have everything in one place, for plotting purposes
                model.discriminator = m3_model.discriminator

            # ---- train a plain classifier on imputed data
            if args.model not in ['m1', 'm2', 'm3', 'gb']:

                # ---- create tf datasets with labels
                train_dataset = utils.get_tf_dataset((Xtrain_i, Strain, ytrain), **config['training'])
                val_dataset = utils.get_tf_dataset((Xval_i, Sval, yval), batch_size=Xval_i.shape[0])

                estimator = MaximumLikelihood()
                estimator.init_tensorboard(os.path.join(EXPERIMENT, args.model))

                optimizer = tf.keras.optimizers.Adam(1e-4)
                trainer.train(model,
                              estimator,
                              optimizer,
                              train_dataset,
                              val_dataset,
                              **config['training'])

                # ---- load best set of weights ( accuracy or p(y|x) ? )
                model.load_weights(estimator.save_dir)

            # ---- train a GB classifier
            if args.model == 'gb':

                print(model)
                model.fit(np.concatenate([Xtrain_nan, Xval_nan], axis=0), np.concatenate([ytrain, yval], axis=0))

            # ---- plot imputed data
            plot_data((Xtrain_i, Strain, ytrain), os.path.join(ASSETS_DIR, prefix + '_data'))
            plot_imputed_data((Xtrain_i, Strain, ytrain), os.path.join(ASSETS_DIR, prefix + '_imputed_data'))
            if args.model == 'learnable-imputation':
                plot_imputed_data_li((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix + '_imputed_data'))

            # ---- plot decision surface
            plot_surface((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix + '_surface'))

            # ---- for m1-m3
            if args.model in ['m1', 'm2', 'm3']:
                plot_classprobs((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix + '_class_probs'))
                plot_posterior((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix))
                plot_single((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix + '_single'))
                plot_multiple((Xtrain_i, Strain, ytrain), model, os.path.join(ASSETS_DIR, prefix + '_multiple'))
                n_latent = config[args.model]['model_settings']['n_latent']
                plot_predictive(model, n_latent, os.path.join(ASSETS_DIR, prefix + '_predictive'))

            # ---- compute performance
            if args.model == 'm3':
                model = m3_model
                ytrain_pred, ytrain_probs = estimator.predict((Xtrain_i, np.ones_like(Xtrain_i).astype(np.float32)), model)
                ytest_pred, ytest_probs = estimator.predict((Xtest_i, np.ones_like(Xtest_i).astype(np.float32)), model)
            elif args.model == 'gb':
                ytrain_pred = model.predict(Xtrain_nan)
                ytrain_probs = model.predict_proba(Xtrain_nan)
                ytest_pred = model.predict(Xtest_nan)
                ytest_probs = model.predict_proba(Xtest_nan)
            else:
                ytrain_pred, ytrain_probs = estimator.predict((Xtrain_i, Strain), model, n_samples=10000, pareto=False)
                ytest_pred, ytest_probs = estimator.predict((Xtest_i, Stest), model, n_samples=10000, pareto=False)

            train_acc, train_acc_on_missing, train_acc_on_observed = \
                utils.get_accs(ytrain, ytrain_pred, Strain)
            test_acc, test_acc_on_missing, test_acc_on_observed = \
                utils.get_accs(ytest, ytest_pred, Stest)

            train_lpyx, train_lpyx_on_missing, train_lpyx_on_observed = \
                utils.get_lpyx(ytrain, ytrain_probs, Strain)
            test_lpyx, test_lpyx_on_missing, test_lpyx_on_observed = \
                utils.get_lpyx(ytest, ytest_probs, Stest)
            # Note, you can compare lpyx based on snis_probs to snis_lpyx:
            # loss, metrics, outputs = estimator(model, (Xtrain_i, Strain, ytrain), n_samples=100)

            # ---- log
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            metrics = {"model": args.model,
                       "m": m,
                       "train acc": train_acc,
                       "test acc": test_acc,
                       "train acc on missing": train_acc_on_missing,
                       "train acc on observed": train_acc_on_observed,
                       "test acc on missing": test_acc_on_missing,
                       "test acc on observed": test_acc_on_observed,
                       "train $\log p(y|x)$": train_lpyx,
                       "test $\log p(y|x)$": test_lpyx,
                       "train $\log p(y|x)$ on missing": train_lpyx_on_missing,
                       "train $\log p(y|x)$ on observed": train_lpyx_on_observed,
                       "test $\log p(y|x)$ on missing": test_lpyx_on_missing,
                       "test $\log p(y|x)$ on observed": test_lpyx_on_observed,
                       "timestamp": current_time}
            results = results.append(pd.DataFrame(metrics, index=[1]), ignore_index=True)
            results.to_pickle(results_file)


if __name__ == '__main__':

    dataset_map = {
        'half-moons': make_moons,
        'circles': make_circles,
        'pin-wheel': make_pinwheel,
        'burger': make_burger
    }

    imputer_map = {
        'supMIWAE': SimpleImputer(strategy='constant', fill_value=0),
        'MIWAE': SimpleImputer(strategy='constant', fill_value=0),
        '0-impute': SimpleImputer(strategy='constant', fill_value=0),
        'learnable-imputation': SimpleImputer(strategy='constant', fill_value=0),
        'permutation-invariance': SimpleImputer(strategy='constant', fill_value=0),
        'PPCA': PPCA(),
        'MICE': IterativeImputer(estimator=BayesianRidge(), max_iter=100, sample_posterior=False, random_state=0),
        'missForest': IterativeImputer(estimator=RandomForestRegressor(n_estimators=100), max_iter=100, random_state=0),
        'GB': SimpleImputer(strategy='constant', fill_value=0),
    }

    model_map = {
        'supMIWAe': MLPMIWAE,
        'MIWAE': MLPMIWAE,
        '0-impute': SimpleDiscriminatorModel,
        'learnable-imputation': LearnableImputationModel,
        'permutation-invariance': PermutationInvarianceModel,
        'PPCA': SimpleDiscriminatorModel,
        'MICE': SimpleDiscriminatorModel,
        'missForest': SimpleDiscriminatorModel,
        'GB': HistGradientBoostingClassifier,
    }

    estimator_map = {
        'supMIWAE': VariationalInference,
        'MIWAE': MaximumLikelihood,
        '0-impute': MaximumLikelihood,
        'learnable-imputation': MaximumLikelihood,
        'permutation-invariance': MaximumLikelihood,
        'PPCA': MaximumLikelihood,
        'MICE': MaximumLikelihood,
        'missForest': MaximumLikelihood,
        'GB': [],
    }

    parser = argparse.ArgumentParser(description='Sklearn dataset examples')
    parser.add_argument('--model', type=str, default="MIWAE", choices=model_map.keys())
    parser.add_argument('--dataset', type=str, default="circles", choices=dataset_map.keys())
    parser.add_argument('--gpu', type=str, default="")
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--config', type=str, default="configs/sklearn/task01.yaml")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    main(args)
