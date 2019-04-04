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
from wats.ext_boxplot import compute_extended_boxplot_stats, plot_extended_boxplot

BOXPLOT_VAR_NAMES = [
    'pressure',
    'geopotential',
    'theta',
    'TKE',
]

KL_DIV_VAR_NAMES = [
    'pressure',
    'geopt',
    'theta',
    'ua',
    'va',
    'wa',
    'QVAPOR'
]

KL_DIV_VAR_LABELS = [
    r'$p$',
    r'$\phi$',
    r'$\theta$',
    r'$u$',
    r'$v$',
    r'$w$',
    r'$q_{v}\,$'
]

def normalise(arr: np.array) -> np.array:
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    return norm

def compute_boxplot_stats(arr: np.array, label=None) -> dict:
    boxplot_stats = cb.boxplot_stats(arr, whis=[2, 98], labels=[label])
    assert len(boxplot_stats) == 1
    boxplot_stats = boxplot_stats[0]
    # There may be many outliers very close together.
    # This increases memory usage for plotting considerably and increases data size.
    # Let's remove all duplicates that we don't need.
    boxplot_stats['fliers'] = np.unique(boxplot_stats['fliers'].round(decimals=5))
    return boxplot_stats

def compute_and_append_stats(ref_dir: Path, trial_dir: Path, out_dir: Path) -> None:
    logging.info('Reading reference and trial data for analysis')

    var_ref_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES}
    var_trial_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES}
    rel_errs = []
    boxplot_stats_per_file = {}
    ext_boxplot_stats_per_file = {}
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

        boxplot_stats_per_var = []
        ext_boxplot_stats_per_var = []
        for var_name in BOXPLOT_VAR_NAMES:
            logging.info(f'  Summary statistics: reading {var_name} & computing relative error')
            var_ref = read_var(nc_ref, var_name)
            var_trial = read_var(nc_trial, var_name)
            small = np.count_nonzero(np.abs(var_ref) < 0.01)
            if small > 0:
                logging.warn('  Found {} ref values < 0.01. Min: {}, Max: {}'.format(
                    small, np.min(var_ref), np.max(var_ref)))
            rel_err = calc_rel_error(var_ref, var_trial)
            rel_err = rel_err.ravel()
            rel_errs.append(rel_err)
            logging.info(f'  Summary statistics: computing per-quantity boxplot stats')
            boxplot_stats_per_var.append(compute_boxplot_stats(rel_err, label=var_name))
            ext_boxplot_stats_per_var.append(compute_extended_boxplot_stats(rel_err, label=var_name))
        boxplot_stats_per_file[str(rel_path)] = boxplot_stats_per_var
        ext_boxplot_stats_per_file[str(rel_path)] = ext_boxplot_stats_per_var

    rel_errs = np.concatenate(rel_errs)

    logging.info('All data read')

    logging.info('Computing Kullback-Leibler divergence')
    kl_divs = []
    bin_count = 100
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

    logging.info('Computing boxplot stats (combined)')
    boxplot_stats = compute_boxplot_stats(rel_errs)

    logging.info('Computing extended boxplot stats (combined)')
    ext_boxplot_stats = compute_extended_boxplot_stats(rel_errs)

    logging.info('Storing stats')
    stats_path = out_dir / 'stats.pkl'
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as fp:
            stats = pickle.load(fp)
    else:
        stats = []

    trial_name = trial_dir.name
    stats.append((trial_name,
                  boxplot_stats, boxplot_stats_per_file,
                  ext_boxplot_stats, ext_boxplot_stats_per_file,
                  kl_divs))

    with open(stats_path, 'wb') as fp:
        pickle.dump(stats, fp)

def plot(stats_dir: Path, out_dir: Path) -> None:
    stats_path = stats_dir / 'stats.pkl'
    boxplot_path = out_dir / 'boxplot.png'
    ext_boxplot_path = out_dir / 'ext_boxplot.png'
    kl_div_path = out_dir / 'kl_div.png'

    logging.info('Loading stats')
    with open(stats_path, 'rb') as fp:
        stats = pickle.load(fp)

    def parse_trial_name(name: str) -> Tuple[str, str, str, str]:
        parts = name.split('_', maxsplit=4) # Restrict for case of 'dm_sm'
        return {'os': parts[1], 'build_system': parts[2], 'build_type': parts[3], 'mode': parts[4]}

    trial_names = [s[0] for s in stats]
    boxplot_stats_all_trials = [s[1] for s in stats]
    boxplot_stats_all_trials_per_file = [s[2] for s in stats]
    ext_boxplot_stats_all_trials = [s[3] for s in stats]
    ext_boxplot_stats_all_trials_per_file = [s[4] for s in stats]
    kl_divs_all_trials = np.asarray([s[5] for s in stats])

    for trial_idx, trial_name in enumerate(trial_names):
        print(f'{trial_idx}: {trial_name}')

    logging.info('Creating boxplots (per-trial)')
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(10,6),  dpi=300)
    boxplot_ax.set_xlabel('Trial number')
    boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
    sns.despine(boxplot_fig)
    boxplot_ax.bxp(boxplot_stats_all_trials)
    boxplot_ax.set_xticklabels(range(len(trial_names)))
    boxplot_fig.savefig(boxplot_path)
    plt.close(boxplot_fig)

    logging.info('Creating boxplots (per-trial per-file per-quantity)')
    for trial_idx, boxplot_stats_per_file in enumerate(boxplot_stats_all_trials_per_file):
        for rel_path, boxplot_stats_all_vars in boxplot_stats_per_file.items():
            boxplot_fig, boxplot_ax = plt.subplots(figsize=(10,6))
            trial_name = trial_names[trial_idx].replace('wats_', '')
            boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
            boxplot_ax.set_xlabel('Quantity')
            boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
            sns.despine(boxplot_fig)
            boxplot_ax.bxp(boxplot_stats_all_vars)
            clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
            boxplot_path = out_dir / 'boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
            boxplot_fig.savefig(boxplot_path)
            plt.close(boxplot_fig)

    # plot each quantity in a single plot for even greater detail
    for trial_idx, boxplot_stats_per_file in enumerate(boxplot_stats_all_trials_per_file):
        for rel_path, boxplot_stats_all_vars in boxplot_stats_per_file.items():
            for boxplot_stats in boxplot_stats_all_vars:
                boxplot_fig, boxplot_ax = plt.subplots(figsize=(10,6))
                trial_name = trial_names[trial_idx].replace('wats_', '')
                boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                boxplot_ax.set_xlabel('Quantity')
                boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
                sns.despine(boxplot_fig)
                boxplot_ax.bxp([boxplot_stats])
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                quantity_name = boxplot_stats['label']
                boxplot_path = out_dir / 'boxplot_{}_{}_{}.png'.format(trial_name, clean_rel_path, quantity_name)
                boxplot_fig.savefig(boxplot_path)
                plt.close(boxplot_fig)

    logging.info('Creating extended boxplots (per-trial)')
    ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6),  dpi=300)
    ext_boxplot_ax.set_xlabel('Trial number')
    ext_boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
    sns.despine(ext_boxplot_fig)
    plot_extended_boxplot(ext_boxplot_ax, ext_boxplot_stats_all_trials)
    ext_boxplot_ax.set_xticklabels(range(len(trial_names)))
    ext_boxplot_fig.savefig(ext_boxplot_path)
    plt.close(ext_boxplot_fig)

    logging.info('Creating extended boxplots (per-trial per-file per-quantity)')
    for trial_idx, ext_boxplot_stats_per_file in enumerate(ext_boxplot_stats_all_trials_per_file):
        for rel_path, ext_boxplot_stats_all_vars in ext_boxplot_stats_per_file.items():
            ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6))
            trial_name = trial_names[trial_idx].replace('wats_', '')
            ext_boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
            ext_boxplot_ax.set_xlabel('Quantity')
            ext_boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
            sns.despine(ext_boxplot_fig)
            plot_extended_boxplot(ext_boxplot_ax, ext_boxplot_stats_all_vars)
            clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
            ext_boxplot_path = out_dir / 'ext_boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
            ext_boxplot_fig.savefig(ext_boxplot_path)
            plt.close(ext_boxplot_fig)

    # plot each quantity in a single plot for even greater detail
    for trial_idx, ext_boxplot_stats_per_file in enumerate(ext_boxplot_stats_all_trials_per_file):
        for rel_path, ext_boxplot_stats_all_vars in ext_boxplot_stats_per_file.items():
            for ext_boxplot_stats in ext_boxplot_stats_all_vars:
                ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6))
                trial_name = trial_names[trial_idx].replace('wats_', '')
                ext_boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                ext_boxplot_ax.set_xlabel('Quantity')
                ext_boxplot_ax.set_ylabel(r'$\mathbf{\eta}$' + '/1')
                sns.despine(ext_boxplot_fig)
                plot_extended_boxplot(ext_boxplot_ax, [ext_boxplot_stats])
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                quantity_name = ext_boxplot_stats['label']
                ext_boxplot_path = out_dir / 'ext_boxplot_{}_{}_{}.png'.format(trial_name, clean_rel_path, quantity_name)
                ext_boxplot_fig.savefig(ext_boxplot_path)
                plt.close(ext_boxplot_fig)

    logging.info('Creating Kullback-Leibler divergence plot')
    assert kl_divs_all_trials.shape == (len(trial_names), len(KL_DIV_VAR_NAMES)), \
        '{} != {}'.format(kl_divs_all_trials.shape, (len(trial_names), len(KL_DIV_VAR_NAMES)))
    if len(trial_names) > 1:
        # normalise each quantity to [0,1]
        for quantity_idx in range(len(KL_DIV_VAR_NAMES)):
            kl_divs_all_trials[:,quantity_idx] = normalise(kl_divs_all_trials[:,quantity_idx])
    else:
        logging.info('  skipping per-quantity KL normalisation because only one trial given')

    def get_scatter_style(trial: dict) -> Tuple[str,str,float]:
        color = {'Linux': '#33a02c', 'macOS': '#fb9a99', 'Windows': '#1f78b4'}
        marker = {'serial': 'o', 'smpar': '^', 'dmpar': 's', 'dm_sm': 'D'}
        linewidth = {'Debug': 0, 'Release': 1}
        return color[trial['os']], marker[trial['mode']], linewidth[trial['build_type']]

    # We want to have 2 labels for each quantity and plot them next to each other
    # to differentiate more easily between the Make and CMake varant.
    kl_div_var_labels = []
    for name in KL_DIV_VAR_LABELS:
        kl_div_var_labels.append(name + r'$_{\mathrm{Make}}$')
        kl_div_var_labels.append(name + r'$_{\mathrm{CMake}}$')

    kl_fig, kl_ax = plt.subplots(figsize=(10,6), dpi=300)
    for trial_idx, kl_divs in enumerate(kl_divs_all_trials):
        trial = parse_trial_name(trial_names[trial_idx])

        # To plot Make and CMake next to each other for each variable
        kl_divs_make_cmake = np.empty(len(kl_div_var_labels))
        kl_divs_make_cmake.fill(np.nan)
        if trial['build_system'] == 'Make':
            kl_divs_make_cmake[::2] = kl_divs[:]
        if trial['build_system'] == 'CMake':
            kl_divs_make_cmake[1::2] = kl_divs[:]

        color, marker, linewidth = get_scatter_style(trial)
        kl_ax.scatter(kl_div_var_labels, kl_divs_make_cmake, label=trial_idx,
                s=150, edgecolors='k', linewidth=linewidth, c=color, marker=marker, alpha=0.5)
        # FIXME: at the top of each quantity we should show the max un-normalized KL value (in nats)
        if trial_idx in range(0,len(kl_div_var_labels),2):
            kl_ax.annotate(r'$\nabla_{\mathrm{KL\, max =\,}}$' + f'{trial_idx} FIXME', (trial_idx, 1.05))
    sns.despine(kl_fig)
    kl_ax.set_xlabel('Quantity')
    kl_ax.set_ylabel(r'$\hat{\nabla}_{\mathrm{KL}}$'+ '/1')
    kl_fig.savefig(kl_div_path, dpi=200)
    plt.close(kl_fig)

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
