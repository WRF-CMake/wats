# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

import os
import shutil
from pathlib import Path
import platform
import subprocess

def link_file(src_path: Path, link_path: Path) -> None:
    assert src_path.is_file()
    if link_path.exists():
        link_path.unlink()
    try:
        # Windows: requires admin rights, but not restricted to same drive
        link_path.symlink_to(src_path)
    except:
        # Windows: does not require admin rights, but restricted to same drive
        os.link(str(src_path), str(link_path))

def link_folder(src_path: Path, link_path: Path) -> None:
    assert src_path.is_dir()
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