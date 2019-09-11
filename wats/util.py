# WRF-CMake Automated Testing Suite (WATS) (https://github.com/WRF-CMake/wats).
# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

import os
import shutil
from pathlib import Path
import platform
import subprocess
import logging

import colorlog
import colorama

SUCCESS_LOG_LEVEL = 21

def link_file(src_path: Path, link_path: Path) -> None:
    if link_path.exists():
        link_path.unlink()
    try:
        # Windows: requires admin rights, but not restricted to same drive
        link_path.symlink_to(src_path)
    except:
        # Windows: does not require admin rights, but restricted to same drive
        os.link(str(src_path), str(link_path))

def link_folder(src_path: Path, link_path: Path) -> None:
    if platform.system() == 'Windows':
        # Directory junctions don't require admin rights.
        subprocess.check_output('cmd /c mklink /j "{}" "{}"'.format(link_path, src_path), shell=True)
    else:
        link_path.symlink_to(src_path, target_is_directory=True)

def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        link_file(src, dst)
    else:
        link_folder(src, dst)

def get_log_level(equal: bool):
    if equal:
        return SUCCESS_LOG_LEVEL
    else:
        return logging.ERROR

def init_logging():
    colorama.init()
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(SUCCESS_LOG_LEVEL, 'SUCCESS')
    logger = logging.getLogger()
    logger.handlers[0].setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(message)s', datefmt='%H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'reset',
            'SUCCESS':  'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }))