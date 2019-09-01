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
from wats.nccmp import (
    read_var, calc_rel_error, calc_rel_error_range_normalised,
    calc_rel_error_iqr_normalised, calc_range, calc_iqr)
from wats.ext_boxplot import compute_extended_boxplot_stats, plot_extended_boxplot
from wats.latex import abs_err_to_latex

VAR_NAMES = [
    'pressure',
    'geopt',
    'theta',
    'ua',
    'va',
    'wa',
]

VAR_LABELS = [
    r'$p$',
    r'$\phi$',
    r'$\theta$',
    r'$u$',
    r'$v$',
    r'$w$',
]

VAR_UNITS = [
    r'$\mathsf{Pa}$',
    r'$\mathsf{m^2\ s^{-2}}$',
    r'$\mathsf{K}$',
    r'$\mathsf{m\ s^{-1}}$',
    r'$\mathsf{m\ s^{-1}}$',
    r'$\mathsf{m\ s^{-1}}$',
]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def abserr(predictions, targets):
    err = np.abs(predictions - targets)
    return {'mean': err.mean(), 'std': err.std(), 'min': err.min(), 'max': err.max()}

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

    var_ref_all = {var_name: [] for var_name in VAR_NAMES} # type: Dict[str,List[np.array]]
    var_trial_all = {var_name: [] for var_name in VAR_NAMES} # type: Dict[str,List[np.array]]
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

        for var_name in VAR_NAMES:
            logging.info(f'  Reading {var_name}')
            var_ref = read_var(nc_ref, var_name, time_idx)
            var_trial = read_var(nc_trial, var_name, time_idx)
            var_ref_all[var_name].append(var_ref.ravel())
            var_trial_all[var_name].append(var_trial.ravel())

        boxplot_stats_per_var = []
        ext_boxplot_stats_per_var = []
        for var_name in VAR_NAMES:
            logging.info(f'  Summary statistics: reading {var_name} & computing relative error')
            var_ref = read_var(nc_ref, var_name, time_idx)
            var_trial = read_var(nc_trial, var_name, time_idx)

            rel_err = calc_rel_error_range_normalised(var_ref, var_trial)
            
            rel_err = rel_err.ravel()
            rel_errs.append(rel_err)
            logging.info(f'  Summary statistics: computing {var_name} boxplot stats')
            boxplot_stats_per_var.append(compute_boxplot_stats(rel_err, label=var_name))
            ext_boxplot_stats = compute_extended_boxplot_stats(rel_err, label=var_name)
            ext_boxplot_stats_per_var.append(ext_boxplot_stats)

        boxplot_stats_per_file[str(rel_path)] = boxplot_stats_per_var
        ext_boxplot_stats_per_file[str(rel_path)] = ext_boxplot_stats_per_var

    rel_errs = np.concatenate(rel_errs)

    logging.info('All data read')

    logging.info('Computing per-quantity statistics')
    pearson_coeffs = []
    rmses = []
    maes = []
    ae_stds = []
    ae_mins = []
    ae_maxs = []
    iqrs = []
    means = []
    boxplot_stats_refs = []
    boxplot_stats_trials = []
    ranges = []
    bin_count = 100
    for var_name in VAR_NAMES:
        logging.info(f'  Processing {var_name}')
        ref_concat = np.concatenate(var_ref_all[var_name])
        trial_concat = np.concatenate(var_trial_all[var_name])
        # Pearson
        pearson_coeff = scipy.stats.pearsonr(ref_concat, trial_concat)[0]
        pearson_coeffs.append(pearson_coeff)
        # RMSE
        rmse_ = rmse(ref_concat, trial_concat)
        rmses.append(rmse_)
        # Absolute error (mean, stddev, min, max)
        abserr_ = abserr(ref_concat, trial_concat)
        maes.append(abserr_['mean'])
        ae_stds.append(abserr_['std'])
        ae_mins.append(abserr_['min'])
        ae_maxs.append(abserr_['max'])
        # Inter-quartile range of data
        iqr = calc_iqr(ref_concat)
        iqrs.append(iqr)
        # Mean of data
        mean = np.mean(ref_concat)
        means.append(mean)
        # Range of data
        ranges.append(calc_range(ref_concat))
        # Boxplot stats of data
        boxplot_stats_trial = compute_boxplot_stats(trial_concat)
        boxplot_stats_trials.append(boxplot_stats_trial)
        boxplot_stats_ref = compute_boxplot_stats(ref_concat)
        boxplot_stats_refs.append(boxplot_stats_ref)

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
                  pearson_coeffs, rmses,
                  maes, ae_stds, ae_mins, ae_maxs,
                  iqrs, means, boxplot_stats_refs, boxplot_stats_trials,
                  ranges))

    with open(stats_path, 'wb') as fp:
        pickle.dump(stats, fp)

def plot(stats_path: Path, plots_dir: Path, trial_filter: Optional[str]=None, detailed=False, dpi=200) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if detailed:
        plots_detailed_dir = plots_dir / 'detailed'
        plots_detailed_dir.mkdir(exist_ok=True)
    
    exts = ['png', 'svg', 'pdf']

    def savefig(fig, path):
        for ext in exts:
            fig.savefig(str(path).format(ext=ext))

    quantity_labels_path = plots_dir / 'quantity_labels.csv'
    quantity_units_path = plots_dir / 'quantity_units.csv'
    trial_labels_path = plots_dir / 'trial_labels.csv'
    ref_boxplot_path = plots_dir / 'ref_boxplot.{ext}'
    trial_boxplot_path = plots_dir / 'trial_boxplot.{ext}'
    rel_err_boxplot_path = plots_dir / 'rel_err_boxplot.{ext}'
    rel_err_ext_boxplot_path = plots_dir / 'rel_err_ext_boxplot.{ext}'
    rel_err_ext_boxplot_vert_path = plots_dir / 'rel_err_ext_boxplot_vert.{ext}'
    rel_err_ext_boxplot_test_path = plots_dir / 'rel_err_ext_boxplot_test.{ext}'
    pearson_path = plots_dir / 'pearson.{ext}'
    rmse_path = plots_dir / 'rmse.{ext}'
    mae_path = plots_dir / 'mae.{ext}'
    ae_std_path = plots_dir / 'ae_std.{ext}'
    ae_min_path = plots_dir / 'ae_min.{ext}'
    ae_max_path = plots_dir / 'ae_max.{ext}'
    ae_tex_path = plots_dir / 'ae.tex'
    nrmse_mean_path = plots_dir / 'nrmse_mean.{ext}'
    nrmse_range_path = plots_dir / 'nrmse_range.{ext}'
    nrmse_iqr_path = plots_dir / 'nrmse_iqr.{ext}'
    mean_path = plots_dir / 'mean.{ext}'
    range_path = plots_dir / 'range.{ext}'
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
    rel_err_boxplot_stats_all_trials = [s[1] for s in stats]
    rel_err_boxplot_stats_all_trials_per_file = [s[2] for s in stats]
    rel_err_ext_boxplot_stats_all_trials = [s[3] for s in stats]
    rel_err_ext_boxplot_stats_all_trials_per_file = [s[4] for s in stats]
    pearson_coeffs_all_trials = np.asarray([s[5] for s in stats])
    rmses_all_trials = np.asarray([s[6] for s in stats])
    maes_all_trials = np.asarray([s[7] for s in stats])
    ae_std_all_trials = np.asarray([s[8] for s in stats])
    ae_min_all_trials = np.asarray([s[9] for s in stats])
    ae_max_all_trials = np.asarray([s[10] for s in stats])
    iqrs_all_trials = np.asarray([s[11] for s in stats])
    means_all_trials = np.asarray([s[12] for s in stats])
    ref_boxplot_stats_all_trials = [s[13] for s in stats]
    trial_boxplot_stats_all_trials = [s[14] for s in stats]
    ranges_all_trials = np.asarray([s[15] for s in stats])

    def rel_to_pct_error(stats):
        for n in ['mean', 'med', 'q1', 'q3', 'cilo', 'cihi', 'whislo', 'whishi', 'fliers']:
            stats[n] *= 100

    def rel_to_percent_error_ext(stats):
        for n in ['median', 'mean', 'percentiles', 'min', 'max', 'values_min', 'values_max']:
            stats[n] *= 100

    for stats in rel_err_boxplot_stats_all_trials:
        rel_to_pct_error(stats)
    for stats_per_file in rel_err_boxplot_stats_all_trials_per_file:
        for stats_per_var in stats_per_file.values():
            for stats in stats_per_var:
                rel_to_pct_error(stats)
    for stats in rel_err_ext_boxplot_stats_all_trials:
        rel_to_percent_error_ext(stats)
    for stats_per_file in rel_err_ext_boxplot_stats_all_trials_per_file:
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

    logging.info('Saving trial labels')
    with open(trial_labels_path, 'w') as fp:
        fp.write(','.join(trial_labels))

    logging.info('Saving quantity labels')
    with open(quantity_labels_path, 'w') as fp:
        fp.write(','.join(VAR_LABELS))
    
    logging.info('Saving quantity units')
    with open(quantity_units_path, 'w') as fp:
        fp.write(','.join(VAR_UNITS))

    logging.info('Creating ref boxplots (per-quantity)')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ref_boxplot_stats_per_quantity = ref_boxplot_stats_all_trials[0]
    for i, (var_label, var_unit, ref_boxplot_stats) \
          in enumerate(zip(VAR_LABELS, VAR_UNITS, ref_boxplot_stats_per_quantity)):
        sub_ax = fig.add_subplot(len(VAR_NAMES), 1, i + 1)
        sub_ax.bxp([ref_boxplot_stats], vert=False)
        sub_ax.set_yticklabels([f'{var_label} in {var_unit}'])
    ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, ref_boxplot_path)
    plt.close(fig)

    logging.info('Creating rel err boxplots (per-trial)')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ax.set_ylabel('Trial')
    ax.set_xlabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(fig)
    ax.bxp(rel_err_boxplot_stats_all_trials, vert=False)
    ax.set_yticklabels(trial_labels)
    fig.tight_layout()
    savefig(fig, rel_err_boxplot_path)
    plt.close(fig)

    if detailed:
        logging.info('Creating boxplots (per-trial per-file per-quantity)')
        for trial_name, boxplot_stats_per_file in zip(trial_names, rel_err_boxplot_stats_all_trials_per_file):
            trial_name = trial_name.replace('wats_', '')
            for rel_path, boxplot_stats_all_vars in boxplot_stats_per_file.items():
                fig, ax = plt.subplots(figsize=(10,6))
                ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                ax.set_xlabel('Quantity')
                ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
                sns.despine(fig)
                ax.bxp(boxplot_stats_all_vars)
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                rel_err_boxplot_path = plots_detailed_dir / 'boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
                savefig(fig, rel_err_boxplot_path)
                plt.close(fig)

    logging.info('Creating extended boxplots (per-trial)')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ax.set_ylabel('Trial')
    ax.set_xlabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(fig)
    plot_extended_boxplot(ax, rel_err_ext_boxplot_stats_all_trials,
                          offscale_minmax=offscale_minmax, vert=False,
                          showmeans=False)
    ax.set_yticklabels(trial_labels)
    fig.tight_layout()
    savefig(fig, rel_err_ext_boxplot_path)
    plt.close(fig)

    logging.info('Creating extended boxplots (per-trial) -- vertical')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    ax.set_xlabel('Trial number')
    ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
    sns.despine(fig)
    plot_extended_boxplot(ax, rel_err_ext_boxplot_stats_all_trials,
                          offscale_minmax=offscale_minmax, vert=True)
    ax.set_xticklabels(trial_idxs)
    fig.tight_layout()
    savefig(fig, rel_err_ext_boxplot_vert_path)
    plt.close(fig)

    logging.info('Creating extended boxplot -- for legend')
    stats = dict(
        median=0,
        mean=0,
        percentiles=[-40, -30, -20, -10, 10, 20, 30, 40],
        min=-60,
        max=60,
        label=''
        )
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    plot_extended_boxplot(ax, [stats]*20, showmeans=False,
                          offscale_minmax=False, vert=True)
    savefig(fig, rel_err_ext_boxplot_test_path)
    plt.close(fig)

    if detailed:
        logging.info('Creating extended boxplots (per-trial per-file per-quantity)')
        for trial_name, ext_boxplot_stats_per_file in zip(trial_names, rel_err_ext_boxplot_stats_all_trials_per_file):
            trial_name = trial_name.replace('wats_', '')
            for rel_path, ext_boxplot_stats_all_vars in ext_boxplot_stats_per_file.items():
                fig, ax = plt.subplots(figsize=(10,6))
                ax.set_title('Trial: {}\nFile: {}'.format(trial_name, rel_path))
                ax.set_xlabel('Quantity')
                ax.set_ylabel(r'$\mathbf{\delta}$' + ' in %')
                sns.despine(fig)
                plot_extended_boxplot(ax, ext_boxplot_stats_all_vars,
                                      offscale_minmax=offscale_minmax)
                clean_rel_path = rel_path.replace('/', '_').replace('\\', '_')
                rel_err_ext_boxplot_path = plots_detailed_dir / 'ext_boxplot_{}_{}.png'.format(trial_name, clean_rel_path)
                savefig(fig, rel_err_ext_boxplot_path)
                plt.close(fig)

    logging.info('Creating Pearson correlation coefficient heatmap plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(pearson_coeffs_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, pearson_path)
    plt.close(fig)

    logging.info('Creating RMSE table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, rmse_path)
    plt.close(fig)

    logging.info('Creating MAE table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(maes_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, mae_path)
    plt.close(fig)
    np.savetxt(str(mae_path).format(ext='csv'), maes_all_trials, fmt='%.18f')

    logging.info('Creating absolute error stddev table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(ae_std_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, ae_std_path)
    plt.close(fig)
    np.savetxt(str(ae_std_path).format(ext='csv'), ae_std_all_trials, fmt='%.18f')

    logging.info('Creating absolute error min table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(ae_min_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, ae_min_path)
    plt.close(fig)
    np.savetxt(str(ae_min_path).format(ext='csv'), ae_min_all_trials, fmt='%.18f')

    logging.info('Creating absolute error max table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(ae_max_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, ae_max_path)
    plt.close(fig)
    np.savetxt(str(ae_max_path).format(ext='csv'), ae_max_all_trials, fmt='%.18f')

    logging.info('Creating absolute error latex table')
    tex = abs_err_to_latex(trial_labels, [f'{l} in {u}' for l,u in zip(VAR_LABELS, VAR_UNITS)],
                           maes_all_trials, ae_std_all_trials, ae_max_all_trials)
    with open(ae_tex_path, 'w') as fp:
        fp.write(tex)

    logging.info('Creating means table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(means_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, mean_path)
    plt.close(fig)
    np.savetxt(str(mean_path).format(ext='csv'), means_all_trials, fmt='%.18f')

    logging.info('Creating ranges table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(ranges_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, range_path)
    plt.close(fig)
    np.savetxt(str(range_path).format(ext='csv'), ranges_all_trials, fmt='%.18f')

    logging.info('Creating IQR table plot')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(iqrs_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar=False, cmap=['white'], linewidths=.5, linecolor='k',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, iqr_path)
    plt.close(fig)
    np.savetxt(str(iqr_path).format(ext='csv'), iqrs_all_trials, fmt='%.18f')

    logging.info('Creating NRMSE (mean-normalised) heatmap')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials / means_all_trials * 100, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar_kws={'label': 'NRMSPE in %'}, cmap='viridis',
                ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Trial')
    fig.tight_layout()
    savefig(fig, nrmse_mean_path)
    plt.close(fig)
    
    logging.info('Creating NRMSE (range-normalised) heatmap')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials / ranges_all_trials * 100, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar_kws={'label': 'NRMSPE in %'}, cmap='viridis',
                ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Trial')
    fig.tight_layout()
    savefig(fig, nrmse_range_path)
    plt.close(fig)

    logging.info('Creating NRMSE (IQR-normalised) heatmap')
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    sns.heatmap(rmses_all_trials / iqrs_all_trials, annot=True, fmt='.3g',
                xticklabels=VAR_LABELS, yticklabels=trial_labels,
                cbar_kws={'label': 'NRMSPE in %'}, cmap='viridis',
                ax=ax)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, nrmse_iqr_path)
    plt.close(fig)

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
    plot_parser.add_argument('--detailed', action='store_true',
                             help='Whether to produce additional plots per-file per-quantity')

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
        plot(stats_path, args.plots_dir, args.trial_filter, args.detailed, args.dpi)
    else:
        assert False
