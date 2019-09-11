# WRF-CMake Automated Testing Suite (WATS) (https://github.com/WRF-CMake/wats).
# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Union
import os
import platform
import tarfile
from pathlib import Path
import shutil
import logging
import tempfile

import requests

GEO_DATA_URL = 'http://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz'
MET_DATA_URL = 'http://www2.mmm.ucar.edu/wrf/TUTORIAL_DATA/colorado_march16.tar.gz'
MET_VTABLE = 'Vtable.GFS'

def download_met_data(out_path: Union[str,Path]) -> None:
    download_and_extract(MET_DATA_URL, out_path)

def download_geo_data(out_path: Union[str,Path]) -> None:
    download_and_extract(GEO_DATA_URL, out_path)

def download_and_extract(url: str, out_path: Union[str,Path]) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        return
    Path.mkdir(out_path, parents=True)
    logging.info('Downloading ' + url)
    response = requests.get(url)
    tmp_path = tempfile.mktemp(prefix='wats', suffix='.tar.gz')
    try:
        with open(tmp_path, 'wb') as f:
            f.write(response.content)
        logging.info('Extracting to {}'.format(out_path))
        if platform.system() == 'Windows':
            # The orogwd* datasets contain a folder with name 'con' which
            # is reserved on Windows and has to be handled specially.
            # Note that the extracted 'con' folder cannot be accessed or deleted from
            # Windows Explorer. It can be deleted from the command line
            # with `rd /q /s \\?\c:\path\to\geog\dataset\con`.
            windows_extract_with_reserved_names(str(tmp_path), str(out_path))
        else:
            shutil.unpack_archive(str(tmp_path), str(out_path))
    finally:
        os.remove(tmp_path)

def windows_extract_with_reserved_names(tar_path: str, dst_path: str) -> None:
    ''' 
    This function extracts tar archives that can contain the reserved folder name
    'con' at the last hierarchy level.
    See https://stackoverflow.com/a/50810859.
    '''
    CON = 'con' # reserved name on Windows
    dst_path = os.path.abspath(dst_path)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.replace('/', '\\')
            path = os.path.join(dst_path, name)
            if member.isdir():
                if os.path.basename(name) == CON:
                    path = r'\\?' + '\\' + path
                os.mkdir(path)
            elif member.isfile():
                if os.path.dirname(name) == CON:
                    path = r'\\?' + '\\' + path
                with open(path, 'wb') as fp:
                    shutil.copyfileobj(tar.extractfile(member), fp)
            else:
                raise RuntimeError('unsupported tar item type')

if __name__ == '__main__':
    download_met_data('geo')
    download_met_data('met')
