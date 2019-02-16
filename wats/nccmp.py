# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple, Union
import logging
from collections import namedtuple

import numpy as np
from numpy import ma
import netCDF4 as nc

from wats.util import get_log_level

WRF_NODATA = 32768.0

WRF_CATEGORICAL = set([
    'LANDMASK',
    'LAKEMASK',
    'LANDSEA',
    'XLAND',
    'LU_INDEX',
    'SCB_DOM',
    'SCT_DOM',
    'ISLTYP',
    'IVGTYP',
    'NEST_POS'
])

Stats = namedtuple('Stats',
    ['equal', # whether differences are within tolerance
     'max_abs_diff', 'max_rel_diff', 'mean_abs_diff', 'mean_rel_diff', # continuous variable
     'category_mismatch_ratio']) # categorical variable 

# Merged stats don't have absolute values due to the different magnitudes
# of different variables.
MergedStats = namedtuple('MergedStats',
    ['equal',
     'max_rel_diff', 'mean_rel_diff',
     'max_category_mismatch_ratio'])

def compare_categorical_vars(var1: np.array, var2: np.array, name: str, tol: float) -> Stats:
    mismatches = np.count_nonzero(var1 != var2)
    ratio = mismatches / var1.size
    equal = ratio <= tol
    stats = Stats(equal, 0, 0, 0, 0, ratio)
    if ratio > 0:
        logging.log(get_log_level(stats), 
            "Diff for {} ({}D): cat_mismatch={:.4f}% ({} of {} pixels) {}".format(
                name, np.squeeze(var1).ndim,
                ratio*100, mismatches, var1.size,
                ('' if equal else ' -> ABOVE THRESHOLD')))
    return stats

def compare_continuous_vars(var1: np.array, var2: np.array, name: str, tol: float, mean: bool) -> Stats:
    var1 = ma.masked_equal(var1, WRF_NODATA)
    var2 = ma.masked_equal(var2, WRF_NODATA)

    assert (var1.mask == var2.mask).all()
    assert not var1.mask.all()

    abs_diff = abs(var1 - var2)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()

    denom = max(abs(var1).max(), abs(var2).max())
    if denom > 0:
        rel_diff = abs_diff / denom
        max_rel_diff = max_abs_diff / denom
    else:
        rel_diff = np.zeros_like(var1)
        max_rel_diff = 0.0

    mean_rel_diff = rel_diff.mean()
    
    if mean:
        equal = mean_rel_diff <= tol
        extra = ''
    else:
        above_thresh = rel_diff > tol 
        above_thresh_count = np.count_nonzero(above_thresh)
        equal = above_thresh_count == 0
        extra = '' if equal else ' ({} pixels)'.format(above_thresh_count)

    stats = Stats(equal, max_abs_diff, max_rel_diff, mean_abs_diff, mean_rel_diff, 0)

    if max_abs_diff > 0:
        logging.log(get_log_level(stats), 
            "Diff for {} ({}D): max_abs={:.2e} max_rel={:.2e} mean_abs={:.2e} mean_rel={:.2e}{}".format(
                name, np.squeeze(var1).ndim,                
                max_abs_diff, max_rel_diff, mean_abs_diff, mean_rel_diff,
                ('' if equal else ' -> ABOVE THRESHOLD') + extra))
    
    return stats

def compare_vars(nc1: nc.Dataset, nc2: nc.Dataset, name: str, tol: float, mean=False) -> Stats:
    var1 = nc1.variables[name][:]
    var2 = nc2.variables[name][:]

    if var1.shape != var2.shape:
        dims = nc1.variables[name].dimensions
        raise RuntimeError(f'Shape mismatch for {name}: {var1.shape} != {var2.shape} ({dims})')
    
    if not np.issubdtype(var1.dtype, np.number):
        if (var1 != var2).any():
            raise RuntimeError(f'Non-numeric mismatch for {name}: {var1} != {var2}')
        return Stats(True, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif name in WRF_CATEGORICAL:
        return compare_categorical_vars(var1, var2, name, tol)
    else:
        return compare_continuous_vars(var1, var2, name, tol, mean)

def compare(path1: str, path2: str, tol: float, mean=False) -> MergedStats:
    nc1 = nc.Dataset(path1, 'r')
    nc2 = nc.Dataset(path2, 'r')

    var_names = set(nc1.variables.keys()).union(nc2.variables.keys())

    file_stats = MergedStats(True, 0.0, 0.0, 0.0)
    for var_name in sorted(var_names):
        var_stats = compare_vars(nc1, nc2, var_name, tol, mean)
        file_stats = merge_stats(file_stats, var_stats)
    return file_stats

def merge_stats(stats1: MergedStats, stats2: Union[Stats,MergedStats]) -> MergedStats:
    try:
        ratio = stats2.category_mismatch_ratio
    except AttributeError:
        ratio = stats2.max_category_mismatch_ratio
    return MergedStats(
        stats1.equal and stats2.equal,
        max(stats1.max_rel_diff, stats2.max_rel_diff),
        max(stats1.mean_rel_diff, stats2.mean_rel_diff),
        max(stats1.max_category_mismatch_ratio, ratio))

def is_identical(stats: MergedStats) -> bool:
    return stats.max_rel_diff == 0 and stats.max_category_mismatch_ratio == 0