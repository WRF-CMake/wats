# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple
import logging
from collections import namedtuple

import numpy as np
from numpy import ma
import netCDF4 as nc

WRF_NODATA = 32768.0

Stats = namedtuple('Stats', ['equal', 'max_abs_diff', 'max_rel_diff', 'mean_abs_diff', 'mean_rel_diff'])

def compare_vars(nc1: nc.Dataset, nc2: nc.Dataset, name: str, tol: float, relative=False, mean=False) -> Stats:
    var1 = nc1.variables[name][:]
    var2 = nc2.variables[name][:]

    if var1.shape != var2.shape:
        dims = nc1.variables[name].dimensions
        raise RuntimeError(f'Shape mismatch for {name}: {var1.shape} != {var2.shape} ({dims})')
    
    if not np.issubdtype(var1.dtype, np.number):
        equal = (var1 == var2).all()
        diff = 0.0 if equal else 1.0
        stats = Stats(equal, diff, diff, diff, diff)
        return stats

    var1 = ma.masked_equal(var1, WRF_NODATA)
    var2 = ma.masked_equal(var2, WRF_NODATA)

    assert var1.mask == var2.mask
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
        if relative:
            equal = mean_rel_diff <= tol
        else:
            equal = mean_abs_diff <= tol
        extra = ''
    else:
        if relative:
            above_thresh = rel_diff > tol
        else:
            above_thresh = abs_diff > tol    
        above_thresh_count = np.count_nonzero(above_thresh)
        equal = above_thresh_count == 0
        extra = '' if equal else ' ({} pixels)'.format(above_thresh_count)

    logging.info("Diff for {}: max_abs={:.2e} max_rel={:.2e} mean_abs={:.2e} mean_rel={:.2e}{}".format(name,
        max_abs_diff, max_rel_diff, mean_abs_diff, mean_rel_diff,
        ('' if equal else ' -> ABOVE THRESHOLD') + extra))
    
    stats = Stats(equal, max_abs_diff, max_rel_diff, mean_abs_diff, mean_rel_diff)
    return stats

def compare(path1: str, path2: str, tol: float, relative=False, mean=False) -> Stats:
    nc1 = nc.Dataset(path1, 'r')
    nc2 = nc.Dataset(path2, 'r')

    var_names = set(nc1.variables.keys()).union(nc2.variables.keys())

    file_stats = Stats(True, 0.0, 0.0, 0.0, 0.0)
    for var_name in sorted(var_names):
        var_stats = compare_vars(nc1, nc2, var_name, tol, relative, mean)
        file_stats = merge_stats(file_stats, var_stats)
    return file_stats

def merge_stats(stats1: Stats, stats2: Stats) -> Stats:
    return Stats(
        stats1.equal and stats2.equal,
        max(stats1.max_abs_diff, stats2.max_abs_diff),
        max(stats1.max_rel_diff, stats2.max_rel_diff),
        max(stats1.mean_abs_diff, stats2.mean_abs_diff),
        max(stats1.mean_rel_diff, stats2.mean_rel_diff))
