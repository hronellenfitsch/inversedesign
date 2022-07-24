#!/usr/bin/env python3

"""
  parallel.py

  generates data using parallel processing
"""

import networks
import spectral

import numpy as np
import scipy as sp
import networkx as nx

import pickle

from joblib import Parallel, delayed

def optimize_mid(size, gap_fracs):
    triang = networks.TriangularGrid2D(n=size, x_periodic=True, y_periodic=True)

    opt = spectral.PointMassOptimizerMid(triang, gap_fracs)
    return opt.optimize(), gap_fracs, triang

def generate(size, N_gaps, N_sims):
    np.random.seed(42)
    k = 0

    for n_gaps in range(1, N_gaps+1):
        for n_sim in range(N_sims):
            gap_fracs = np.random.random(n_gaps)

            yield delayed(optimize_mid)(size, gap_fracs)

def generate_fixed(size, N_sims):
    np.random.seed(42)
    k = 0

    for n_sim in range(N_sims):
        gap_fracs = [0.75]

        yield delayed(optimize_mid)(size, gap_fracs)

if __name__ == '__main__':
    size = 12

    N_sims = 100
    N_gaps = 4

    results = []
    x0s = []
    gaps = []

    # results = Parallel(n_jobs=4, verbose=50)(job for job in generate(size, N_gaps, N_sims))
    results = Parallel(n_jobs=3, verbose=50)(job for job in generate_fixed(size, N_sims))

    print(results)

    with open('data/size_{}_N_{}_0.75.pkl'.format(size, N_sims), 'wb') as f:
        pickle.dump(results, f)
