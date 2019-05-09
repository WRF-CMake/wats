# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple, Dict, List, Optional
import os
import sys
import glob
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
    #'QVAPOR'
]

KL_DIV_VAR_LABELS = [
    r'$p$',
    r'$\phi$',
    r'$\theta$',
    r'$u$',
    r'$v$',
    r'$w$',
    #r'$q_{v}\,$'
]

def normalise(arr: np.array) -> np.array:
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    return norm

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def compute_boxplot_stats(arr: np.array, label=None) -> dict:
    boxplot_stats = cb.boxplot_stats(arr, whis=1.5, labels=[label])
    assert len(boxplot_stats) == 1
    boxplot_stats = boxplot_stats[0]
    # There may be many outliers very close together.
    # This increases memory usage for plotting considerably and increases data size.
    # Let's remove all duplicates that we don't need.
    boxplot_stats['fliers'] = np.unique(boxplot_stats['fliers'].round(decimals=5))
    return boxplot_stats

def compute_and_append_stats(ref_dir: Path, trial_dir: Path, stats_path: Path,
                             path_filter: Optional[str]=None, time_idx: Optional[int]=None) -> None:
    logging.info('Reading reference and trial data for analysis')
    logging.info('Reference: {}'.format(ref_dir))
    logging.info('Trial: {}'.format(trial_dir))

    var_ref_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES} # type: Dict[str,List[np.array]]
    var_trial_all = {var_name: [] for var_name in KL_DIV_VAR_NAMES} # type: Dict[str,List[np.array]]
    rel_errs = []
    boxplot_stats_per_file = {}
    ext_boxplot_stats_per_file = {}
    for ref_path in ref_dir.glob('wrf/*/wrfout_*'):
        rel_path = ref_path.relative_to(ref_dir)
        trial_path = trial_dir / rel_path

        if path_filter and path_filter not in str(rel_path):
            continue

        logging.info(f'Processing {rel_path}')

        nc_ref = nc.Dataset(ref_path, 'r')
        nc_trial = nc.Dataset(trial_path, 'r')

        for var_name in KL_DIV_VAR_NAMES:
            logging.info(f'  Kullback-Leibler divergence: reading {var_name}')
            var_ref = read_var(nc_ref, var_name, time_idx)
            var_trial = read_var(nc_trial, var_name, time_idx)
            var_ref_all[var_name].append(var_ref.ravel())
            var_trial_all[var_name].append(var_trial.ravel())

        boxplot_stats_per_var = []
        ext_boxplot_stats_per_var = []
        for var_name in BOXPLOT_VAR_NAMES:
            logging.info(f'  Summary statistics: reading {var_name} & computing relative error')
            var_ref = read_var(nc_ref, var_name, time_idx)
            var_trial = read_var(nc_trial, var_name, time_idx)
            small = np.count_nonzero(np.abs(var_ref) < 0.01)
            if small > 0:
                logging.warn('  Found {} ref values < 0.01. Min: {}, Max: {}'.format(
                    small, np.min(var_ref), np.max(var_ref)))
            rel_err = calc_rel_error(var_ref, var_trial)
            large_err = np.abs(rel_err) > 0.5
            if large_err.any():
                logging.warn('  Found relative error > 0.5')
                logging.warn('    Ref: {}'.format(var_ref[large_err].ravel()))
                logging.warn('    Trial: {}'.format(var_trial[large_err].ravel()))
                if var_name == 'TKE':
                    for sub_var_name in ['U', 'V', 'W']:
                        dims = nc_ref.dimensions
                        bottom_top = dims['bottom_top'].size
                        south_north = dims['south_north'].size
                        west_east = dims['west_east'].size
                        sub_var_ref = read_var(nc_ref, sub_var_name, time_idx)
                        sub_var_trial = read_var(nc_trial, sub_var_name, time_idx)
                        sub_var_ref = sub_var_ref[:,:bottom_top,:south_north,:west_east]
                        sub_var_trial = sub_var_trial[:,:bottom_top,:south_north,:west_east]
                        logging.warn('    Ref [{}]: {}'.format(sub_var_name, sub_var_ref[large_err].ravel()))
                        logging.warn('    Trial [{}]: {}'.format(sub_var_name, sub_var_trial[large_err].ravel()))
            rel_err = rel_err.ravel()
            rel_errs.append(rel_err)
            logging.info(f'  Summary statistics: computing {var_name} boxplot stats')
            boxplot_stats_per_var.append(compute_boxplot_stats(rel_err, label=var_name))
            ext_boxplot_stats = compute_extended_boxplot_stats(rel_err, label=var_name)
            ext_boxplot_stats_per_var.append(ext_boxplot_stats)
            values_min = ext_boxplot_stats['values_min']
            values_max = ext_boxplot_stats['values_max']
            values_min_sample = np.sort(values_min)[:1000]
            values_max_sample = np.sort(values_max)[::-1][:1000]
            logging.info('  99.9%-100%: {}'.format(values_max_sample))
            if len(values_max_sample) < len(values_max):
                logging.info('  (truncated to 1000, total: {})'.format(len(values_max)))
            logging.info('  0%-0.1%: {}'.format(values_min_sample))
            if len(values_min_sample) < len(values_min):
                logging.info('  (truncated to 1000, total: {})'.format(len(values_min)))

        boxplot_stats_per_file[str(rel_path)] = boxplot_stats_per_var
        ext_boxplot_stats_per_file[str(rel_path)] = ext_boxplot_stats_per_var

    rel_errs = np.concatenate(rel_errs)

    logging.info('All data read')

    logging.info('Computing Kullback-Leibler divergence, Pearson correlation coefficients, RMSE')
    kl_divs = []
    pearson_coeffs = []
    rmses = []
    iqrs = []
    means = []
    bin_count = 100
    for var_name in KL_DIV_VAR_NAMES:
        logging.info(f'  Processing {var_name}')
        ref_concat = np.concatenate(var_ref_all[var_name])
        trial_concat = np.concatenate(var_trial_all[var_name])
        # Pearson
        pearson_coeff = scipy.stats.pearsonr(ref_concat, trial_concat)[0]
        pearson_coeffs.append(pearson_coeff)
        # KL
        hist_min = np.min([ref_concat, trial_concat])
        hist_max = np.max([ref_concat, trial_concat])
        ref_freq, _ = np.histogram(ref_concat, bins=bin_count, range=(hist_min, hist_max), density=True)
        trial_freq, _ = np.histogram(trial_concat, bins=bin_count, range=(hist_min, hist_max), density=True)
        kl_div = scipy.stats.entropy(trial_freq, ref_freq)
        kl_divs.append(kl_div)
        # RMSE
        rmse_ = rmse(ref_concat, trial_concat)
        rmses.append(rmse_)
        # Inter-quartile range
        q1, q3 = np.percentile(ref_concat, [25, 75])
        iqr = q3 - q1
        iqrs.append(iqr)
        # Mean
        mean = np.mean(ref_concat)
        means.append(mean)

    logging.info('Computing boxplot stats (combined)')
    boxplot_stats = compute_boxplot_stats(rel_errs)

    logging.info('Computing extended boxplot stats (combined)')
    ext_boxplot_stats = compute_extended_boxplot_stats(rel_errs)

    logging.info('Storing stats')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as fp:
            stats = pickle.load(fp)
    else:
        stats = []

    trial_name = trial_dir.name
    stats.append((trial_name,
                  boxplot_stats, boxplot_stats_per_file,
                  ext_boxplot_stats, ext_boxplot_stats_per_file,
                  kl_divs, pearson_coeffs, rmses, iqrs, means))

    with open(stats_path, 'wb') as fp:
        pickle.dump(stats, fp)

def plot(stats_path: Path, plots_dir: Path, trial_filter: Optional[str]=None, detailed=False, dpi=200) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if detailed:
        plots_detailed_dir = plots_dir / 'detailed'
        plots_detailed_dir.mkdir(exist_ok=True)
    
    exts = ['png', 'svg']

    def savefig(fig, path):
        for ext in exts:
            fig.savefig(str(path).format(ext=ext))

    boxplot_path = plots_dir / 'boxplot.{ext}'
    ext_boxplot_path = plots_dir / 'ext_boxplot.{ext}'
    ext_boxplot_vert_path = plots_dir / 'ext_boxplot_vert.{ext}'
    ext_boxplot_test_path = plots_dir / 'ext_boxplot_test.{ext}'
    kl_div_path = plots_dir / 'kl_div.{ext}'
    pearson_path = plots_dir / 'pearson.{ext}'
    rmse_path = plots_dir / 'rmse.{ext}'
    nrmse_path = plots_dir / 'nrmse.{ext}'
    rmseiqr_path = plots_dir / 'rmseiqr.{ext}'
    mean_path = plots_dir / 'mean.{ext}'
    iqr_path = plots_dir / 'iqr.{ext}'


    offscale_minmax = True

    logging.info('Loading stats')
    with open(stats_path, 'rb') as fp:
        stats = pickle.load(fp)

    all_trial_idxs = list(range(len(stats)))

    if trial_filter:
        trial_idxs = [i for i in all_trial_idxs if trial_filter in stats[i][0]]
        assert len(trial_idxs) > 0
        stats = [stats[i] for i in trial_idxs]
    else:
        trial_idxs = all_trial_idxs

    def parse_trial_name(name: str) -> dict:
        parts = name.split('_', maxsplit=4) # Restrict for case of 'dm_sm'
        return {'os': parts[1], 'build_system': parts[2], 'build_type': parts[3], 'mode': parts[4]}

    trial_names = [s[0] for s in stats]
    boxplot_stats_all_trials = [s[1] for s in stats]
    boxplot_stats_all_trials_per_file = [s[2] for s in stats]
    ext_boxplot_stats_all_trials = [s[3] for s in stats]
    ext_boxplot_stats_all_trials_per_file = [s[4] for s in stats]
    kl_divs_all_trials = np.asarray([s[5] for s in stats])
    pearson_coeffs_all_trials = np.asarray([s[6] for s in stats])
    rmses_all_trials = np.asarray([s[7] for s in stats])
    iqrs_all_trials = np.asarray([s[8] for s in stats])
    means_all_trials = np.asarray([s[9] for s in stats])

    def rel_to_pct_error(stats):
        for n in ['mean', 'med', 'q1', 'q3', 'cilo', 'cihi', 'whislo', 'whishi', 'fliers']:
            stats[n] *= 100

    def rel_to_percent_error_ext(stats):
        for n in ['median', 'mean', 'percentiles', 'min', 'max', 'values_min', 'values_max']:
            stats[n] *= 100

    for stats in boxplot_stats_all_trials:
        rel_to_pct_error(stats)
    for stats_per_file in boxplot_stats_all_trials_per_file:
        for stats_per_var in stats_per_file.values():
            for stats in stats_per_var:
                rel_to_pct_error(stats)
    for stats in ext_boxplot_stats_all_trials:
        rel_to_percent_error_ext(stats)
    for stats_per_file in ext_boxplot_stats_all_trials_per_file:
        for stats_per_var in stats_per_file.values():
            for stats in stats_per_var:
                rel_to_percent_error_ext(stats)

    for trial_idx, trial_name in zip(trial_idxs, trial_names):
        print(f'{trial_idx}: {trial_name}')

    trial_labels = []
    for trial_name in trial_names:
        trial = parse_trial_name(trial_name)
        trial_labels.append('{os}/{system}/{type}/{mode}'.format(
            os=trial['os'], system=trial['build_system'],
            type=trial['build_type'], mode=trial['mode']))

    logging.info('Creating boxplots (per-trial)')
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    boxplot_ax.set_ylabel('Trial')
    boxplot_ax.set_xlabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(boxplot_fig)
    boxplot_ax.bxp(boxplot_stats_all_trials, vert=False)
    #boxplot_ax.set_xticklabels(trial_idxs)
    boxplot_ax.set_yticklabels(trial_labels)
    boxplot_fig.tight_layout()
    savefig(boxplot_fig, boxplot_path)
    plt.close(boxplot_fig)

    if detailed:
        logging.info('Creating boxplots (per-trial per-file per-quantity)')
        for trial_name, boxplot_stats_per_file in zip(trial_names, boxplot_stats_all_trials_per_file):
            trial_name = trial_name.replace('wats_', '')
            for rel_path, boxplot_stats_all_vars in boxplot_stats_per_file.items():
                boxplot_fig, boxplot_ax = plt.subplots(figsize=(10,6))
                boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                boxplot_ax.set_xlabel('Quantity')
                boxplot_ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
                sns.despine(boxplot_fig)
                boxplot_ax.bxp(boxplot_stats_all_vars)
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                boxplot_path = plots_detailed_dir / 'boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
                savefig(boxplot_fig, boxplot_path)
                plt.close(boxplot_fig)

    logging.info('Creating extended boxplots (per-trial)')
    ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ext_boxplot_ax.set_ylabel('Trial')
    ext_boxplot_ax.set_xlabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(ext_boxplot_fig)
    plot_extended_boxplot(ext_boxplot_ax, ext_boxplot_stats_all_trials,
                          offscale_minmax=offscale_minmax, vert=False,
                          showmeans=False)
    ext_boxplot_ax.set_yticklabels(trial_labels)
    ext_boxplot_fig.tight_layout()
    savefig(ext_boxplot_fig, ext_boxplot_path)
    plt.close(ext_boxplot_fig)

    logging.info('Creating extended boxplots (per-trial) -- vertical')
    ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ext_boxplot_ax.set_xlabel('Trial number')
    ext_boxplot_ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(ext_boxplot_fig)
    plot_extended_boxplot(ext_boxplot_ax, ext_boxplot_stats_all_trials,
                          offscale_minmax=offscale_minmax, vert=True)
    ext_boxplot_ax.set_xticklabels(trial_idxs)
    ext_boxplot_fig.tight_layout()
    savefig(ext_boxplot_fig, ext_boxplot_vert_path)
    plt.close(ext_boxplot_fig)

    logging.info('Creating extended boxplot -- for legend')
    stats = dict(
        median=0,
        mean=0,
        percentiles=[-40, -30, -20, -10, 10, 20, 30, 40],
        min=-60,
        max=60,
        label=''
        )
    ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    plot_extended_boxplot(ext_boxplot_ax, [stats]*20, showmeans=False,
                          offscale_minmax=False, vert=True)
    savefig(ext_boxplot_fig, ext_boxplot_test_path)
    plt.close(ext_boxplot_fig)

    if detailed:
        logging.info('Creating extended boxplots (per-trial per-file per-quantity)')
        for trial_name, ext_boxplot_stats_per_file in zip(trial_names, ext_boxplot_stats_all_trials_per_file):
            trial_name = trial_name.replace('wats_', '')
            for rel_path, ext_boxplot_stats_all_vars in ext_boxplot_stats_per_file.items():
                ext_boxplot_fig, ext_boxplot_ax = plt.subplots(figsize=(10,6))
                ext_boxplot_ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                ext_boxplot_ax.set_xlabel('Quantity')
                ext_boxplot_ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
                sns.despine(ext_boxplot_fig)
                plot_extended_boxplot(ext_boxplot_ax, ext_boxplot_stats_all_vars,
                                      offscale_minmax=offscale_minmax)
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                ext_boxplot_path = plots_detailed_dir / 'ext_boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
                savefig(ext_boxplot_fig, ext_boxplot_path)
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

    kl_fig, kl_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    for trial_name, kl_divs in zip(trial_names, kl_divs_all_trials):
        trial = parse_trial_name(trial_name)

        # To plot Make and CMake next to each other for each variable
        kl_divs_make_cmake = np.empty(len(kl_div_var_labels))
        kl_divs_make_cmake.fill(np.nan)
        if trial['build_system'] == 'Make':
            kl_divs_make_cmake[::2] = kl_divs[:]
        if trial['build_system'] == 'CMake':
            kl_divs_make_cmake[1::2] = kl_divs[:]

        color, marker, linewidth = get_scatter_style(trial)
        kl_ax.scatter(kl_div_var_labels, kl_divs_make_cmake,
                s=150, edgecolors='k', linewidth=linewidth, c=color, marker=marker, alpha=0.5)
        # FIXME: at the top of each quantity we should show the max un-normalized KL value (in nats)
        #if trial_idx in range(0,len(kl_div_var_labels),2):
        #    kl_ax.annotate(r'$\nabla_{\mathrm{KL\, max =\,}}$' + f'{trial_idx} FIXME', (trial_idx, 1.05))
    sns.despine(kl_fig)
    kl_ax.set_xlabel('Quantity')
    kl_ax.set_ylabel(r'$\hat{\nabla}_{\mathrm{KL}}$'+ '/1')
    kl_fig.tight_layout()
    savefig(kl_fig, kl_div_path)
    plt.close(kl_fig)

    logging.info('Creating Pearson correlation coefficient heatmap plot')
    pearson_fig, pearson_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(pearson_coeffs_all_trials, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                ax=pearson_ax)
    pearson_fig.tight_layout()
    savefig(pearson_fig, pearson_path)
    plt.close(pearson_fig)

    logging.info('Creating RMSE table plot')
    rmse_fig, rmse_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=rmse_ax)
    rmse_fig.tight_layout()
    savefig(rmse_fig, rmse_path)
    plt.close(rmse_fig)

    logging.info('Creating means table plot')
    mean_fig, mean_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(means_all_trials, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=mean_ax)
    mean_fig.tight_layout()
    savefig(mean_fig, mean_path)
    plt.close(mean_fig)

    logging.info('Creating NRMSE heatmap')
    nrmse_fig, nrmse_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials / means_all_trials * 100, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                cbar_kws={'label': 'NRMSE in %'}, cmap='viridis',
                ax=nrmse_ax)
    nrmse_ax.set_xlabel('Quantity')
    nrmse_ax.set_ylabel('Trial')
    nrmse_fig.tight_layout()
    savefig(nrmse_fig, nrmse_path)
    plt.close(nrmse_fig)

    logging.info('Creating IQR table plot')
    iqr_fig, iqr_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(iqrs_all_trials, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=iqr_ax)
    iqr_fig.tight_layout()
    savefig(iqr_fig, iqr_path)
    plt.close(iqr_fig)

    logging.info('Creating RMSEIQR heatmap')
    rmseiqr_fig, rmseiqr_ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials / iqrs_all_trials, annot=True, fmt='.3g',
                xticklabels=KL_DIV_VAR_LABELS, yticklabels=trial_labels,
                ax=rmseiqr_ax)
    rmseiqr_fig.tight_layout()
    savefig(rmseiqr_fig, rmseiqr_path)
    plt.close(rmseiqr_fig)

if __name__ == '__main__':
    init_logging()

    def as_path(path: str) -> Path:
        return Path(path).absolute()
    
    def as_paths(path_pattern: str) -> List[Path]:
        paths = list(map(as_path, glob.glob(path_pattern)))
        assert len(paths) > 0, 'Invalid path or pattern: {}'.format(path_pattern)
        return paths

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')

    compute_parser = subparsers.add_parser('compute')
    compute_parser.add_argument('ref_dir', type=as_path,
                             help='Input reference data directory')
    compute_parser.add_argument('trial_dirs', type=as_paths, nargs='+',
                             help='Input trial data directories, supports glob patterns and ignores reference directory')
    compute_parser.add_argument('--filter', dest='path_filter', type=str,
                             help='Optional file path filter, e.g. _d01_')
    compute_parser.add_argument('--time-idx', type=int,
                             help='Optional time index filter, e.g. 0 for first timestep only')
    compute_parser.add_argument('--stats-dir', type=as_path, default=ROOT_DIR / 'stats',
                             help='Output statistics directory')
    compute_parser.add_argument('--append', action='store_true',
                             help='Whether to append to existing statistics')
    compute_parser.add_argument('--ref-trial-pairs', action='store_true',
                             help='Whether folders are given as reference/trial pairs')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('--stats-dir', type=as_path, default=ROOT_DIR / 'stats',
                             help='Input statistics directory')
    plot_parser.add_argument('--plots-dir', type=as_path, default=ROOT_DIR / 'plots',
                             help='Output plots directory')
    plot_parser.add_argument('--dpi', type=int, default=200,
                             help='DPI of plots')
    plot_parser.add_argument('--filter', dest='trial_filter', type=str,
                             help='Optional trial name filter, e.g. macOS')
    plot_parser.add_argument('--skip-detailed', action='store_true',
                             help='Whether to skip producing additional plots per-file per-quantity')

    args = parser.parse_args()

    stats_path = args.stats_dir / 'stats.pkl'

    if args.subparser_name == 'compute':
        args.stats_dir.mkdir(parents=True, exist_ok=True)
        if not args.append and stats_path.exists():
            stats_path.unlink()
        trial_dirs = []
        for trial_dirs_ in args.trial_dirs:
            for trial_dir in trial_dirs_:
                if trial_dir != args.ref_dir:
                    trial_dirs.append(trial_dir)
        if args.ref_trial_pairs:
            dirs = [args.ref_dir] + trial_dirs
            assert len(dirs) % 2 == 0
            for i in range(0, len(dirs), 2):
                ref_dir = dirs[i]
                trial_dir = dirs[i+1]
                compute_and_append_stats(ref_dir, trial_dir, stats_path, args.path_filter, args.time_idx)
        else:
            for trial_dir in trial_dirs:
                compute_and_append_stats(args.ref_dir, trial_dir, stats_path, args.path_filter, args.time_idx)
    elif args.subparser_name == 'plot':
        plot(stats_path, args.plots_dir, args.trial_filter, not args.skip_detailed, args.dpi)
    else:
        assert False
