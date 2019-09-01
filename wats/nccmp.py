# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple, Union, List, Set, Optional
import logging
from collections import namedtuple

import numpy as np
from numpy import ma
import netCDF4 as nc
import wrf

from wats.util import get_log_level

def read_var(ds: nc.Dataset, name: str, time_idx: Optional[int]=None) -> np.array:
    if name == 'KE':
        u = wrf.getvar(ds, 'U', time_idx, squeeze=False).values
        v = wrf.getvar(ds, 'V', time_idx, squeeze=False).values
        w = wrf.getvar(ds, 'W', time_idx, squeeze=False).values
        dims = ds.dimensions
        bottom_top = dims['bottom_top'].size
        south_north = dims['south_north'].size
        west_east = dims['west_east'].size
        u = u[:,:bottom_top,:south_north,:west_east]
        v = v[:,:bottom_top,:south_north,:west_east]
        w = w[:,:bottom_top,:south_north,:west_east]
        var = 0.5 * (u**2 + v**2 + w**2)
    else:
        try:
            var = wrf.getvar(ds, name, time_idx, squeeze=False).values
        except:
            var = ds.variables[name][:]
            if time_idx is not None:
                var = var[time_idx:time_idx+1]
    return var

def calc_rel_error(var_ref: np.array, var_trial: np.array) -> np.array:
    ref_zeros = var_ref == 0
    trial_nonzeros_cnt = np.count_nonzero(var_trial[ref_zeros])
    if trial_nonzeros_cnt > 0:
        raise ValueError(f'Reference contains {trial_nonzeros_cnt} points with zero where trial is non-zero')

    with np.errstate(divide='ignore'):
        rel_error = (var_trial - var_ref) / var_ref
    rel_error[ref_zeros] = 0

    return rel_error

def calc_rel_error_range_normalised(var_ref: np.array, var_trial: np.array) -> np.array:
    err = var_trial - var_ref
    ref_range = calc_range(var_ref)

    if ref_range == 0:
        raise ValueError('ref_range == 0')

    rel_error = err / ref_range
    return rel_error

def calc_rel_error_iqr_normalised(var_ref: np.array, var_trial: np.array) -> np.array:
    err = var_trial - var_ref
    ref_iqr = calc_iqr(var_ref)

    if ref_iqr == 0:
        raise ValueError('ref_iqr == 0')

    rel_error = err / ref_iqr
    return rel_error

def calc_range(arr: np.array) -> float:
    range_ = np.max(arr) - np.min(arr)
    return range_

def calc_iqr(arr: np.array) -> float:
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return iqr

def compare_categorical_var(var1: np.array, var2: np.array, name: str, tol_percentage: float) -> bool:
    mismatches = np.count_nonzero(var1 != var2)
    ratio = mismatches / var1.size
    equal = ratio*100 <= tol_percentage
    logging.log(get_log_level(equal), 
        "{}: category mismatches {:.4f}% ({} of {} pixels) {}".format(
            name,
            ratio*100, mismatches, var1.size,
            ('' if equal else ' -> ABOVE THRESHOLD')))
    return equal

def compare_continuous_var(var_ref: np.array, var_trial: np.array, name: str, tol: float, mean: bool) -> bool:
    try:
        rel_error = calc_rel_error(var_ref, var_trial)
    except ValueError as e:
        logging.error('{}: {}'.format(name, e))
        return False

    rel_error_abs = abs(rel_error)

    max_rel_diff = rel_error_abs.max()
    mean_rel_diff = rel_error_abs.mean()
    
    if mean:
        equal = mean_rel_diff <= tol
        extra = ''
    else:
        above_thresh = rel_error_abs > tol
        above_thresh_count = np.count_nonzero(above_thresh)
        equal = above_thresh_count == 0
        extra = '' if equal else ' ({} pixels)'.format(above_thresh_count)

    logging.info(
        '{}: reference mean={:.3e} stddev={:.3e} min={:.3e} max={:.3e}'.format(
            name, var_ref.mean(), var_ref.std(), var_ref.min(), var_ref.max()))

    logging.info(
        '{}: trial     mean={:.3e} stddev={:.3e} min={:.3e} max={:.3e}'.format(
            name, var_trial.mean(), var_trial.std(), var_trial.min(), var_trial.max()))

    logging.log(get_log_level(equal), 
        "{}: relative error max={:.4f}% mean={:.4f}%{}".format(
            name,
            max_rel_diff*100, mean_rel_diff*100,
            ('' if equal else ' -> ABOVE THRESHOLD') + extra))

    return equal

def compare_var(var_ref: np.array, var_trial: np.array, 
                name: str, is_categorical: bool,
                no_data: Optional[Union[float,int]],
                tol: float, mean=False) -> bool:
    if var_ref.shape != var_trial.shape:
        raise RuntimeError(f'Shape mismatch for {name}: {var_ref.shape} != {var_trial.shape}')
    
    is_numeric = np.issubdtype(var_ref.dtype, np.number)

    if no_data is not None and is_numeric:
        var_ref = ma.masked_equal(var_ref, no_data)
        var_trial = ma.masked_equal(var_trial, no_data)
        assert (var_ref.mask == var_trial.mask).all()
        if var_ref.mask.all():
            logging.error('{} has only missing values!')
            return False

    if not is_numeric:
        if (var_ref != var_trial).any():
            logging.error(f'Non-numeric mismatch for {name}: {var_ref} != {var_trial}')
            return False
        return True
    elif is_categorical:
        return compare_categorical_var(var_ref, var_trial, name, tol)
    else:
        return compare_continuous_var(var_ref, var_trial, name, tol, mean)

def compare(path_ref: str, path_trial: str, 
            var_names_categorical: List[str], 
            var_names_continuous: List[str],
            no_data: Optional[Union[float,int]],
            tol_continuous: float, tol_categorical: float,
            mean=False) -> bool:
    nc_ref = nc.Dataset(path_ref, 'r')
    nc_trial = nc.Dataset(path_trial, 'r')

    var_names = {var_name: True for var_name in var_names_categorical}
    var_names.update({var_name: False for var_name in var_names_continuous})

    file_equal = True
    for var_name, is_categorical in sorted(var_names.items()):
        try:
            var_ref = read_var(nc_ref, var_name)
        except Exception as e:
            logging.info(f'"{var_name}" not found or problem opening: {e}. Ignoring.')
            continue
        var_trial = read_var(nc_trial, var_name)

        tol = tol_categorical if is_categorical else tol_continuous

        try:
            var_equal = compare_var(var_ref, var_trial, var_name, is_categorical, no_data, tol, mean)
        except Exception as e:
            logging.error(f'Error processing "{var_name}": {e}', exc_info=e)
            var_equal = False
        file_equal = file_equal and var_equal
    return file_equal
