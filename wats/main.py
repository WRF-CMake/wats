# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Iterable, Union, Tuple, Optional
import os
import sys
import platform
import shutil
from pathlib import Path
import subprocess
import multiprocessing
import time
import argparse
import logging

THIS_DIR = Path(__file__).absolute().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))

import wats.data
from wats.util import link, get_log_level, init_logging
from wats import nccmp

WPS_CASES_DIR = ROOT_DIR / 'cases' / 'wps'
WRF_CASES_DIR = ROOT_DIR / 'cases' / 'wrf'

WRF_NODATA = 32768.0

WPS_CATEGORICAL = set([
    'LANDSEA',
    'LANDMASK', 
    'LU_INDEX',
    'SCB_DOM',
    'SCT_DOM',
])

WRF_CATEGORICAL = set([
    'LANDMASK',
    'LAKEMASK',    
    'XLAND',
    'LU_INDEX',
    'ISLTYP',
    'IVGTYP',
    'NEST_POS',
])

# Continuous variables of ratio type (see https://en.wikipedia.org/wiki/Level_of_measurement).
WPS_CONTINUOUS = set([
    'HGT_M',
    'GHT',
    'HGTMAXW',
    'HGTTROP',
    'SOILHGT',
    'SOILTEMP',
    'PMAXW',
    'PMSL',
    'PSFC',
    'PTROP',
    'SKINTEMP',
    'SNOWH',
    'TT',
    'TTROP'
])

WRF_CONTINUOUS = set([
    'pressure',
    'geopotential',
    'theta',
    'KE'
])

def get_case_name(nml_path: Path) -> str:
    return str(nml_path).split('.')[-1]

def get_run_dir(work_dir: Path) -> Path:
    return work_dir / 'run'

def get_output_dir(work_dir: Path) -> Path:
    return work_dir / 'output'

def create_empty_dir(path: Path) -> None:
    if path.exists():
        logging.info('Removing existing folder {}'.format(path))
        # work-around: handle 'geo' folder link specially as rmtree would recurse into it
        geo_link = path / 'geo'
        if geo_link.exists():
            geo_link.unlink()
        shutil.rmtree(str(path))
    logging.info('Creating folder {}'.format(path))
    path.mkdir(parents=True)

def create_case_dirs(mode: str, nml_path: Path, work_dir: Path) -> Tuple[Path,Path]:
    case_name = get_case_name(nml_path)
    run_dir = get_run_dir(work_dir) / mode / case_name
    output_dir = get_output_dir(work_dir) / mode / case_name
    create_empty_dir(run_dir)
    create_empty_dir(output_dir)
    return run_dir, output_dir

MPI_IMPL = None
MSMPI_ARGS = ['-exitcodes', '-lines']
MPICH_ARGS = ['-print-all-exitcodes']
OPENMPI_ARGS = ['--oversubscribe']

def run_exe(args: Iterable[Union[str, Path]], cwd: Path, use_mpi: bool, retries=3) -> bool:
    args_list = [str(arg) for arg in args]
    tool = Path(args_list[0]).name
    logging.info('Running ' + tool + (' with MPI' if use_mpi else ''))

    if use_mpi:
        if platform.system() == 'Windows':
            # Microsoft MPI
            mpi_path = os.path.join(os.environ['MSMPI_BIN'], 'mpiexec.exe')
            extra_flags = MSMPI_ARGS
        else:
            # MPICH or Open MPI
            mpi_path = 'mpiexec'
            global MPI_IMPL
            if MPI_IMPL is None:
                out = subprocess.check_output([mpi_path, '--version'], universal_newlines=True)
                MPI_IMPL = 'openmpi' if 'open-mpi.org' in out else 'mpich'
                print('Detected ' + MPI_IMPL)
            extra_flags = MPICH_ARGS if MPI_IMPL == 'mpich' else OPENMPI_ARGS
        args_list = [mpi_path, '-n', str(multiprocessing.cpu_count())] + extra_flags + args_list

    success = True
    try:
        result = subprocess.run(args_list, cwd=str(cwd),
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
    except subprocess.CalledProcessError as e:
        success = False
        logging.error(tool + ' failed, output:\n' + e.stdout)
        with (cwd / (tool + '.log')).open('w') as fp:
            fp.write(e.stdout)
        # Try to work around https://github.com/microsoft/azure-pipelines-image-generation/issues/932.
        if 'gethostbyname failed' in e.stdout:
            if retries > 0:
                logging.warn('Sleeping and retrying...')
                time.sleep(30)
                success = run_exe(args, cwd, use_mpi, retries - 1)
    else:
        for err_msg in ['ERROR', 'FATAL', 'Error']:
            if err_msg in result.stdout:
                success = False
                logging.error(tool + 'failed, output:\n' + result.stdout)
                break
        with (cwd / (tool + '.log')).open('w') as fp:
            fp.write(result.stdout)
    return success

def run_wps_case(wps_nml_path: Path, wps_dir: Path, work_dir: Path, use_mpi: bool) -> Path:
    geo_data_dir = work_dir / 'geo'
    met_data_dir = work_dir / 'met'
    wats.data.download_geo_data(geo_data_dir)
    wats.data.download_met_data(met_data_dir)

    run_dir, output_dir = create_case_dirs('wps', wps_nml_path, work_dir)

    link(geo_data_dir, run_dir / 'geo')

    link_grib_py = wps_dir / 'link_grib.py'
    link_grib_cmd = [sys.executable, str(link_grib_py), str(met_data_dir), '-o', str(run_dir)]
    logging.info('Running ' + ' '.join(link_grib_cmd))
    subprocess.run(link_grib_cmd, check=True)

    link(wps_dir / 'geogrid' / 'GEOGRID.TBL.ARW', run_dir / 'geogrid' / 'GEOGRID.TBL')
    link(wps_dir / 'ungrib' / 'Variable_Tables' / wats.data.MET_VTABLE, run_dir / 'Vtable')
    link(wps_dir / 'metgrid' / 'METGRID.TBL.ARW', run_dir / 'metgrid' / 'METGRID.TBL')
    link(wps_nml_path, run_dir / 'namelist.wps')

    success = True
    for tool in ['geogrid.exe', 'ungrib.exe', 'metgrid.exe']:
        supports_mpi = tool != 'ungrib.exe'
        if not run_exe([wps_dir / tool], cwd=run_dir, use_mpi=use_mpi and supports_mpi):
            success = False
            break
    if success:
        logging.info('Executables ran successfully')

    output_patterns = ['geo_em*.nc', 'met_em*.nc', '*.log*']

    for pattern in output_patterns:
        for path in run_dir.glob(pattern):
            shutil.move(str(path), str(output_dir))

    if not success:
        raise RuntimeError('Failure, see above')

    return output_dir

def run_wrf_case(wrf_nml_path: Path, wps_case_output_dir: Path, wrf_dir: Path, work_dir: Path, use_mpi: bool) -> Path:
    run_dir, output_dir = create_case_dirs('wrf', wrf_nml_path, work_dir)

    for path in wps_case_output_dir.glob('met_em*.nc'):
        link(path, run_dir / path.name)

    for path in (wrf_dir / 'test' / 'em_real').iterdir():
        link(path, run_dir / path.name)

    link(wrf_nml_path, run_dir / 'namelist.input')

    success = True
    for tool in ['real.exe', 'wrf.exe']:
        if not run_exe([wrf_dir / 'main' / tool], cwd=run_dir, use_mpi=use_mpi):
            success = False
            break
    if success:
        logging.info('Executables ran successfully')

    output_patterns = ['wrfout*', 'rsl.out.*', 'rsl.error.*']

    for pattern in output_patterns:
        for path in run_dir.glob(pattern):
            shutil.move(str(path), str(output_dir))

    if not success:
        raise RuntimeError('Failure, see above')
    
    return output_dir

def run_cases(mode: str, use_mpi: bool, wrf_dir: Path, wrf_case: Optional[str], wps_dir: Path, wps_case_output_dir: Optional[Path], work_dir: Path) -> None:
    if mode == 'wps':
        for path in sorted(WPS_CASES_DIR.glob('namelist.wps.*')):
            run_wps_case(path, wps_dir, work_dir, use_mpi)
    else:
        if wps_case_output_dir is None:
            wps_nml_path = WPS_CASES_DIR / 'namelist.wps.00'
            logging.info('No WPS output given, running WPS case for {}'.format(wps_nml_path))
            wps_case_output_dir = run_wps_case(wps_nml_path, wps_dir, work_dir, use_mpi)
        else:
            logging.info('Using existing WPS output from {}'.format(wps_case_output_dir))
        for path in sorted(WRF_CASES_DIR.glob('namelist.input.*')):
            case_name = get_case_name(path)
            if wrf_case and wrf_case != case_name:
                continue
            run_wrf_case(path, wps_case_output_dir, wrf_dir, work_dir, use_mpi)

def diff_cases(mode: str, ref_dir: Path, trial_dir: Path,
               tol_continuous: float, tol_categorical: float,
               mean: bool) -> bool:
    aggr_str = 'average' if mean else 'per-pixel'
    logging.info('Comparing {} (reference) <-> {}'.format(ref_dir, trial_dir))
    logging.info('Maximum allowed relative {} difference: {:.4f}%'.format(aggr_str, tol_continuous*100))
    logging.info('Maximum allowed categorical mismatch (percentage of all pixels): {:.4f}%'.format(tol_categorical))   

    exclude_files = ['.git', '.log', 'rsl.']

    if mode == 'wps':
        vars_categorical = WPS_CATEGORICAL
        vars_continuous = WPS_CONTINUOUS
    else:
        vars_categorical = WRF_CATEGORICAL
        vars_continuous = WRF_CONTINUOUS
    
    dir_equal = True
    for ref_path in ref_dir.glob('{}/**/*'.format(mode)):
        if not ref_path.is_file():
            continue
        if any(part in str(ref_path) for part in exclude_files):
            continue
        rel_path = ref_path.relative_to(ref_dir)
        trial_path = trial_dir / rel_path
        logging.info('')
        logging.info('Comparing {}'.format(rel_path))
        
        file_equal = nccmp.compare(ref_path, trial_path,
            vars_categorical, vars_continuous, WRF_NODATA,
            tol_continuous, tol_categorical, mean)

        dir_equal = dir_equal and file_equal
    
    if not dir_equal:
        logging.error('At least one file exceeded the allowed thresholds')
    return dir_equal
    
if __name__ == '__main__':
    init_logging()

    def as_path(path: str) -> Path:
        return Path(path).absolute()

    parser = argparse.ArgumentParser()    
    subparsers = parser.add_subparsers(dest='subparser_name')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--mode', required=True, choices=['wps', 'wrf'], help='whether to run/diff WPS or WRF cases')
    run_parser.add_argument('--mpi', action='store_true', help='whether to use MPI')
    run_parser.add_argument('--wrf-dir', required=True, type=as_path,
                            help='WRF install directory')
    run_parser.add_argument('--wrf-case',
                            help='Run only the given case, e.g. 21')                  
    run_parser.add_argument('--wps-dir', required=True, type=as_path,
                            help='WPS install directory')
    run_parser.add_argument('--wps-case-output-dir', type=as_path,
                            help='For wrf mode, use existing WPS output instead of running WPS')
    run_parser.add_argument('--work-dir', default=ROOT_DIR / 'work', type=as_path,
                            help='Directory to store WRF/WPS output files')

    diff_parser = subparsers.add_parser('diff')
    diff_parser.add_argument('ref_dir', type=as_path,
                             help='Reference output directory')
    diff_parser.add_argument('trial_dir', type=as_path,
                             help='Comparison output directory')
    diff_parser.add_argument('--mode', required=True, choices=['wps', 'wrf'], help='whether to run/diff WPS or WRF cases')
    diff_parser.add_argument('--tol-continuous', default=0.01, type=float,
                            help='Maximum relative error in percent for continuous variables')
    diff_parser.add_argument('--tol-categorical', default=0.01, type=float,
                            help='Maximum percentage mismatch over all points for categorical variables')
    diff_parser.add_argument('--per-pixel-continuous', action='store_true',
                            help='Consider each pixel separate for comparison of continuous variables (default is average over all pixels)')

    args = parser.parse_args()
    if args.subparser_name == 'run':
        run_cases(args.mode, args.mpi, args.wrf_dir, args.wrf_case, args.wps_dir, args.wps_case_output_dir, args.work_dir)
    elif args.subparser_name == 'diff':
        equal = diff_cases(args.mode, args.ref_dir, args.trial_dir, args.tol_continuous/100, args.tol_categorical,
            not args.per_pixel_continuous)
        if not equal:
            sys.exit(1)
    else:
        assert False
