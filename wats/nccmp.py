# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Tuple, Union, List, Set, Optional
import logging
from collections import namedtuple

import numpy as np
from numpy import ma
import netCDF4 as nc
import wrf

from wats.util import get_log_level

def read_var(ds: nc.Dataset, name: str) -> np.array:
    if name == 'TKE':
        u = wrf.getvar(ds, 'U', wrf.ALL_TIMES, squeeze=False).values
        v = wrf.getvar(ds, 'V', wrf.ALL_TIMES, squeeze=False).values
        w = wrf.getvar(ds, 'W', wrf.ALL_TIMES, squeeze=False).values
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
            var = ds.variables[name][:]
        except KeyError:
            var = wrf.getvar(ds, name, wrf.ALL_TIMES)[:]
    return var

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

def compare_continuous_var(var_ref: np.array, var_cmp: np.array, name: str, tol: float, mean: bool) -> bool:
    ref_zeros = var_ref == 0
    cmp_nonzeros_cnt = np.count_nonzero(var_cmp[ref_zeros])
    if cmp_nonzeros_cnt > 0:
        logging.error('{}: reference contains {} points with zero where trial is non-zero -> rel. error undefined!'.format(
            name, cmp_nonzeros_cnt))
        return False

    with np.errstate(divide='ignore'):
        rel_error = (var_ref - var_cmp) / var_ref
    rel_error[ref_zeros] = 0

    max_rel_diff = rel_error.max()
    mean_rel_diff = rel_error.mean()
    
    if mean:
        equal = mean_rel_diff <= tol
        extra = ''
    else:
        above_thresh = rel_error > tol
        above_thresh_count = np.count_nonzero(above_thresh)
        equal = above_thresh_count == 0
        extra = '' if equal else ' ({} pixels)'.format(above_thresh_count)

    logging.info(
        '{}: reference mean={:.3e} stddev={:.3e} min={:.3e} max={:.3e}'.format(
            name, var_ref.mean(), var_ref.std(), var_ref.min(), var_ref.max()))

    logging.info(
        '{}: trial     mean={:.3e} stddev={:.3e} min={:.3e} max={:.3e}'.format(
            name, var_cmp.mean(), var_cmp.std(), var_cmp.min(), var_cmp.max()))

    logging.log(get_log_level(equal), 
        "{}: relative error max={:.4f}% mean={:.4f}%{}".format(
            name,
            max_rel_diff*100, mean_rel_diff*100,
            ('' if equal else ' -> ABOVE THRESHOLD') + extra))

    return equal

def compare_var(var_ref: np.array, var_cmp: np.array, 
                name: str, is_categorical: bool,
                no_data: Optional[Union[float,int]],
                tol: float, mean=False) -> bool:
    if var_ref.shape != var_cmp.shape:
        raise RuntimeError(f'Shape mismatch for {name}: {var_ref.shape} != {var_cmp.shape}')
    
    is_numeric = np.issubdtype(var_ref.dtype, np.number)

    if no_data is not None and is_numeric:
        var_ref = ma.masked_equal(var_ref, no_data)
        var_cmp = ma.masked_equal(var_cmp, no_data)
        assert (var_ref.mask == var_cmp.mask).all()
        if var_ref.mask.all():
            logging.error('{} has only missing values!')
            return False

    if not is_numeric:
        if (var_ref != var_cmp).any():
            logging.error(f'Non-numeric mismatch for {name}: {var_ref} != {var_cmp}')
            return False
        return True
    elif is_categorical:
        return compare_categorical_var(var_ref, var_cmp, name, tol)
    else:
        return compare_continuous_var(var_ref, var_cmp, name, tol, mean)

def compare(path_ref: str, path_cmp: str, 
            var_names_categorical: List[str], 
            var_names_continuous: List[str],
            no_data: Optional[Union[float,int]],
            tol_continuous: float, tol_categorical: float,
            mean=False) -> bool:
    nc_ref = nc.Dataset(path_ref, 'r')
    nc_cmp = nc.Dataset(path_cmp, 'r')

    var_names = {var_name: True for var_name in var_names_categorical}
    var_names.update({var_name: False for var_name in var_names_continuous})

    file_equal = True
    for var_name, is_categorical in sorted(var_names.items()):
        try:
            var_ref = read_var(nc_ref, var_name)
        except Exception as e:
            logging.info(f'"{var_name}" not found or problem opening: {e}. Ignoring.')
            continue
        var_cmp = read_var(nc_cmp, var_name)

        tol = tol_categorical if is_categorical else tol_continuous

        try:
            var_equal = compare_var(var_ref, var_cmp, var_name, is_categorical, no_data, tol, mean)
        except Exception as e:
            logging.error(f'Error processing "{var_name}": {e}', exc_info=e)
            var_equal = False
        file_equal = file_equal and var_equal
    return file_equal
