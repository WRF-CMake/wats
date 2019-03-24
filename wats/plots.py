# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

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

KL_DIV_VAR_NAMES = [
    'pressure',
    'theta',
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
    kl_divs_norm = normalise(np.asarray(kl_divs))

    logging.info('Computing boxplot stats')
    boxplot_stats = cb.boxplot_stats(rel_errs)
    assert len(boxplot_stats) == 1
    boxplot_stats = boxplot_stats[0]
    
    logging.info('Storing stats')
    stats_path = out_dir / 'stats.pkl'
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as fp:
            stats = pickle.load(fp)
    else:
        stats = []
    
    trial_name = trial_dir.name
    stats.append((trial_name, boxplot_stats, kl_divs_norm))
    
    with open(stats_path, 'wb') as fp:
        pickle.dump(stats, fp)

def plot(stats_dir: Path, out_dir: Path) -> None:
    stats_path = stats_dir / 'stats.pkl'
    boxplot_path = out_dir / 'boxplot.png'
    kl_div_path = out_dir / 'kl_div.png'

    with open(stats_path, 'rb') as fp:
        stats = pickle.load(fp)

    logging.info('Creating boxplot')
    boxplot_stats = [s[1] for s in stats]
    boxplot_fig, boxplot_ax = plt.subplots()
    boxplot_ax.set_title('Foo')
    boxplot_ax.bxp(boxplot_stats)
    boxplot_fig.savefig(boxplot_path)

    logging.info('Creating Kullback-Leibler divergence plot')
    kl_divs = [s[2] for s in stats]
    kl_fig, kl_ax = plt.subplots()
    for kl_div in kl_divs:
        kl_ax.scatter(KL_DIV_VAR_NAMES, kl_div)
    kl_fig.savefig(kl_div_path)

if __name__ == '__main__':
    init_logging()

    def as_path(path: str) -> Path:
        return Path(path).absolute()

    parser = argparse.ArgumentParser()    
    subparsers = parser.add_subparsers(dest='subparser_name')

    prepare_parser = subparsers.add_parser('compute')
    prepare_parser.add_argument('ref_dir', type=as_path,
                             help='Reference output directory')
    prepare_parser.add_argument('trial_dir', type=as_path,
                             help='Trial output directory')
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
        compute_and_append_stats(args.ref_dir, args.trial_dir, args.out_dir)
    elif args.subparser_name == 'plot':
        plot(args.stats_dir, args.out_dir)
    else:
        assert False
