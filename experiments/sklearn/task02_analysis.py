import datetime
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns; sns.set()


def mean_and_std(x):
    return '${0:.2f} \pm {1:.2f}$'.format(np.mean(x), np.std(x))


EXPERIMENT = 'sklearn/task02'
RESULTS_DIR = os.path.join('results', EXPERIMENT)
ASSETS_DIR = os.path.join('assets', EXPERIMENT)

# dataset = 'circles'
# dataset = 'half-moons'
# dataset = 'pin-wheel'
dataset = 'burger'

os.makedirs(ASSETS_DIR, exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---- load results
results_files = set(glob.glob(os.path.join(RESULTS_DIR, dataset + '_*.pkl')))
results = pd.DataFrame()
for file in results_files:
    results = results.append(pd.read_pickle(file))

# ---- remove m1 results
results = results.loc[results['model'] != 'm1']

# ---- remove m column
results = results.drop('m', 1)

# ---- update naming
results['model'] = results['model'].replace({'m2': 'supMIWAE',
                                             'm3': 'MIWAE',
                                             'mice': 'MICE',
                                             'ppca-em': 'PPCA',
                                             'gb': 'GB'})

# ---- change model order
order = ['supMIWAE', 'MIWAE', '0-impute', 'learnable-imputation', 'MICE', 'missForest', 'PPCA', 'GB', 'permutation-invariance',]
results['model'] = pd.Categorical(results['model'], order)
results = results.sort_values('model')

# ---- keep cols
cols = [col for col in results.columns if 'test' in col or 'model' in col]

# ---- aggregate results
agg = results.groupby(by=['model'], as_index=False).agg(func=mean_and_std)

# ---- print to latex
with open(os.path.join(ASSETS_DIR, dataset + '_task02_results.tex'), 'w') as f:
    print(agg[cols].to_latex(caption=f"{EXPERIMENT} {dataset} results summarized {current_time}",
                             label=f"tab:{dataset}",
                             escape=False,
                             index=False), file=f)

with open(os.path.join(ASSETS_DIR, dataset + '_task02_results_small.tex'), 'w') as f:
    print(agg[['model', 'test acc', 'test $\log p(y|x)$']].to_latex(caption=f"{EXPERIMENT} {dataset} results summarized {current_time}",
                             label=f"tab:{dataset}_small",
                             escape=False,
                             index=False), file=f)
