# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple
import os
import sys
from pathlib import Path
import argparse
import logging
import pickle

import numpy as np
import scipy.stats
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from matplotlib.markers import MarkerStyle
import seaborn as sns

sns.set_context('paper')
sns.set_style('ticks')

THIS_DIR = Path(__file__).absolute().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))

from wats.util import init_logging
from wats.nccmp import read_var, calc_rel_error

BOXPLOT_VAR_NAMES = [
    'pressure',
    'geopotential',
    'theta',
    'TKE',
]

# TODO: add all continuous vars
KL_DIV_VAR_NAMES = [
    'pressure',
    'theta',
    'ua',
    'va'
]

KL_DIV_VAR_LABELS = [
    'P',
    'T',
    'U',
    'V'
]

def normalise(arr: np.array) -> np.array:
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    return norm

def compute_and_append_stats(ref_dir: Path, trial_dir: Path, out_dir: Path) -> None:
    logging.info('Reading reference and trial data for analysis')

    var_ref_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES}
    var_trial_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES}
    rel_errs = []
    for ref_path in ref_dir.glob('wrf/*/wrfout_*'):
        rel_path = ref_path.relative_to(ref_dir)
        trial_path = trial_dir / rel_path

        logging.info(f'Processing {rel_path}')

        nc_ref = nc.Dataset(ref_path, 'r')
        nc_trial = nc.Dataset(trial_path, 'r')

        for var_name in KL_DIV_VAR_NAMES:
            logging.info(f'  Kullback-Leibler divergence: reading {var_name}')
            var_ref = read_var(nc_ref, var_name)
            var_trial = read_var(nc_trial, var_name)
            var_ref_all[var_name].append(var_ref.ravel())
            var_trial_all[var_name].append(var_trial.ravel())

        for var_name in BOXPLOT_VAR_NAMES:
            logging.info(f'  Summary statistics: reading {var_name} & computing relative error')
            var_ref = read_var(nc_ref, var_name)
            var_trial = read_var(nc_trial, var_name)
            rel_err = calc_rel_error(var_ref, var_trial)
            rel_errs.append(rel_err.ravel())
    rel_errs = np.concatenate(rel_errs)

    logging.info('All data read')

    logging.info('Computing Kullback-Leibler divergence')
    kl_divs = []
    bin_count = 20
    for var_name in KL_DIV_VAR_NAMES:
        logging.info(f'  Processing {var_name}')
        ref_concat = np.concatenate(var_ref_all[var_name])
        trial_concat = np.concatenate(var_trial_all[var_name])
        hist_min = np.min([ref_concat, trial_concat])
        hist_max = np.max([ref_concat, trial_concat])
        ref_freq, _ = np.histogram(ref_concat, bins=bin_count, range=(hist_min, hist_max), density=True)
        trial_freq, _ = np.histogram(trial_concat, bins=bin_count, range=(hist_min, hist_max), density=True)
        kl_div = scipy.stats.entropy(trial_freq, ref_freq)
        kl_divs.append(kl_div)

    logging.info('Computing boxplot stats')
    boxplot_stats = cb.boxplot_stats(rel_errs)
    assert len(boxplot_stats) == 1
    boxplot_stats = boxplot_stats[0]
    # There may be many outliers very close together.
    # This increases memory usage for plotting considerably and increases data size.
    # Let's remove all duplicates that we don't need.
    boxplot_stats['fliers'] = np.unique(boxplot_stats['fliers'].round(decimals=5))

    logging.info('Storing stats')
    stats_path = out_dir / 'stats.pkl'
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as fp:
            stats = pickle.load(fp)
    else:
        stats = []

    trial_name = trial_dir.name
    stats.append((trial_name, boxplot_stats, kl_divs))

    with open(stats_path, 'wb') as fp:
        pickle.dump(stats, fp)

def plot(stats_dir: Path, out_dir: Path) -> None:
    stats_path = stats_dir / 'stats.pkl'
    boxplot_path = out_dir / 'boxplot.png'
    kl_div_path = out_dir / 'kl_div.png'

    logging.info('Loading stats')
    with open(stats_path, 'rb') as fp:
        stats = pickle.load(fp)

    def parse_trial_name(name: str) -> Tuple[str, str, str, str]:
        parts = name.split('_', maxsplit=4) # Restrict for case of 'dm_sm'
        return {'os': parts[1], 'build_system': parts[2], 'build_type': parts[3], 'mode': parts[4]}

    trial_names = [s[0] for s in stats]
    for trial_idx, trial_name in enumerate(trial_names):
        print(f'{trial_idx}: {trial_name}')

    logging.info('Creating boxplot')
    boxplot_stats_all_trials = [s[1] for s in stats]
    boxplot_fig, boxplot_ax = plt.subplots()
    boxplot_ax.set_xlabel('Trial number')
    boxplot_ax.set_ylabel(r'$\eta$' + ' in 1')
    sns.despine(boxplot_fig)
    boxplot_ax.bxp(boxplot_stats_all_trials)
    boxplot_ax.set_xticklabels(range(len(trial_names)))
    boxplot_fig.savefig(boxplot_path, dpi=200)

    logging.info('Creating Kullback-Leibler divergence plot')
    kl_divs_all_trials = np.asarray([s[2] for s in stats])
    assert kl_divs_all_trials.shape == (len(trial_names), len(KL_DIV_VAR_NAMES))
    # normalise each quantity to [0,1]
    for trial_idx in range(len(KL_DIV_VAR_NAMES)):
        kl_divs_all_trials[:,trial_idx] = normalise(kl_divs_all_trials[:,trial_idx])

    def get_scatter_style(trial: dict) -> Tuple[str,str,str,float]:
        fillstyle = {'Make': 'top', 'CMake': 'full'}
        color = {'Linux': '#33a02c', 'macOS': '#fb9a99', 'Windows': '#1f78b4'}
        alpha = {'Debug': 1, 'Release': 0.5}
        marker = {'serial': 'o', 'smpar': '^', 'dmpar': 's', 'dm_sm': 'D'}
        return color[trial['os']], marker[trial['mode']], fillstyle[trial['build_system']], alpha[trial['build_type']]

    kl_fig, kl_ax = plt.subplots()
    for trial_idx, kl_divs in enumerate(kl_divs_all_trials):
        trial = parse_trial_name(trial_names[trial_idx])
        color, marker, fillstyle, alpha = get_scatter_style(trial)
        kl_ax.scatter(KL_DIV_VAR_LABELS, kl_divs, label=trial_idx,
                      s=500, edgecolors='k', linewidth=1, 
                      c=color, marker=MarkerStyle(marker, fillstyle), alpha=alpha)
    sns.despine(kl_fig)
    kl_ax.set_xlabel('Variable name')
    kl_ax.set_ylabel('Normalised KL Divergence in 1')
    kl_fig.savefig(kl_div_path, dpi=200)

if __name__ == '__main__':
    init_logging()

    def as_path(path: str) -> Path:
        return Path(path).absolute()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')

    prepare_parser = subparsers.add_parser('compute')
    prepare_parser.add_argument('ref_dir', type=as_path,
                             help='Reference output directory')
    prepare_parser.add_argument('trial_dirs', type=as_path, nargs='+',
                             help='Trial output directories')
    prepare_parser.add_argument('--out-dir', type=as_path, default=ROOT_DIR / 'stats',
                             help='Output directory for computed statistics')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('--stats-dir', type=as_path, default=ROOT_DIR / 'stats',
                             help='Statistics directory')
    plot_parser.add_argument('--out-dir', type=as_path, default=ROOT_DIR / 'plots',
                             help='Output directory')

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.subparser_name == 'compute':
        for trial_dir in args.trial_dirs:
            compute_and_append_stats(args.ref_dir, trial_dir, args.out_dir)
    elif args.subparser_name == 'plot':
        plot(args.stats_dir, args.out_dir)
    else:
        assert False
