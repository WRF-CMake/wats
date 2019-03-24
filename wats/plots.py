# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

import os
import sys
from pathlib import Path
import argparse
import logging
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).absolute().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))

from wats.util import init_logging

BOXPLOT_VAR_NAMES = [
    'pressure',
    'geopotential',
    'theta',
    'TKE',
]

KL_DIV_VAR_NAMES = [
    'T',
    'P',
]

def plot_partial(ref_dir: Path, cmp_dir: Path, out_dir: Path) -> None:
    partial_path = out_dir / 'partial.pkl'
    boxplot_path = out_dir / 'boxplot.png'
    kl_div_path = out_dir / 'kl_div.png'

    # Read partial plotting data.
    # Note that there is no easy way to draw a boxplot with pre-computed stats.
    # Therefore, to avoid having to keep all data on disk/memory, we literally draw
    # a part of the plot, save the plot object to disk, and continue later again.
    if os.path.exists(partial_path):
        with open(partial_path, 'rb') as fp:
            kl_partial, boxplot_fig, boxplot_ax = pickle.load(fp)
    else:
        kl_partial = []
        boxplot_fig = boxplot_ax = plt.subplots()
        boxplot_ax.set_title('Foo')

    # Compare reference and trial data.
    
    # For KL divergence, per quantity, we
    # (1) flatten and concatenate all values from all cases, then
    # (2) compute histograms for the reference and trial data, and finally
    # (3) compute the entropy between both histograms.
    # As last step, all computed entries are normalised.
    
    # For the boxplot, 
    # TODO continue here

    # Create (partial) plots.
    i = len(kl_partial) - 1
    boxplot_ax.boxplot(boxplot_data, positions=[i])
    boxplot_ax.set_xlim(-0.5, i + 1)
    boxplot_ax.set_xticks(np.arange(i + 1))
    boxplot_ax.set_xticklabels([str(i) for i in range(i + 1)])
    boxplot_fig.savefig(boxplot_path)

    kl_fig, kl_ax = plt.subplots()
    for trial_name, kl_div in kl_partial:
        kl_ax.scatter(KL_DIV_VAR_NAMES, kl_div)
    kl_fig.savefig(kl_div_path)

    # Save partial plotting data.
    with open(partial_path, 'wb') as fp:
        pickle.dump((kl_partial, boxplot_fig, boxplot_ax), fp)

if __name__ == '__main__':
    init_logging()

    def as_path(path: str) -> Path:
        return Path(path).absolute()

    parser = argparse.ArgumentParser()    
    subparsers = parser.add_subparsers(dest='subparser_name')

    diff_parser = subparsers.add_parser('plot-partial')
    diff_parser.add_argument('ref_dir', type=as_path,
                             help='Reference output directory')
    diff_parser.add_argument('cmp_dir', type=as_path,
                             help='Comparison output directory')
    diff_parser.add_argument('--out-dir', type=as_path, default=ROOT_DIR / 'plots',
                             help='Output directory')

    args = parser.parse_args()
    if args.subparser_name == 'plot-partial':
        plot_partial(args.ref_dir, args.cmp_dir, args.out_dir)
    else:
        assert False
