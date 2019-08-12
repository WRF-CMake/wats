# WRF-CMake Automated Testing Suite

The WRF-CMake Automated Testing Suite (WATS) is a command-line tool that runs WRF simulations and compares their outputs between multiple WRF installations. It was developed as part of WRF-CMake to assess whether the CMake-based build system has any influence on the results compared to using the existing build system of hand-crafted Makefiles (WRF-Make). It is used as part of the [Continuous Integration setup](https://dev.azure.com/WRF-CMake/wrf/_build) of WRF-CMake which validates pull requests and all commits to the main development branch. We call these tests *regression tests*. The output of running the tool is a set of plots that are stored as artifact in each successful build. For interpretation of and comparison to reference plots, see the [WRF-CMake paper](https://github.com/openjournals/joss-reviews/issues/1468) (currently in review).

## How it works

Running WATS is split up into two phases:
1. Per WRF compilation/installation, run simulations for multiple cases
2. Compare simulation outputs and create plots

The cases are defined in namelist files (see the [cases/](cases) folder) and aim cover a wide range of conditions.
[Geographical](http://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz) and [meteorological](http://www2.mmm.ucar.edu/wrf/TUTORIAL_DATA/colorado_march16.tar.gz) data is automatically downloaded from UCAR.

## Running locally

In the [Continuous Integration setup](https://dev.azure.com/WRF-CMake/wrf/_build) of WRF-CMake, we perform a series of compilation and regression tests at each commit using WATS on [Windows, macOS, and Linux](https://dev.azure.com/WRF-CMake/wrf/_build).

When you build WRF yourself then you have already done a compilation test. If you like to replicate the regression tests using WATS, then follow the steps below. The steps assume a Linux or macOS system and may have to be modified for Windows.

```sh
git clone https://github.com/WRF-CMake/wats.git

# Install Python packages, either via conda:
conda env create -n wats -f wats/environment.yml
conda activate wats
# Or via pip:
pip install -r wats/requirements.txt

# Run test cases
# Note: Add --mpi if WRF was compiled with MPI support
python wats/wats/main.py run --mode wrf --wrf-dir /path/to/wrf --wps-dir /path/to/wps
# Move to target folder
# Naming scheme: wats_{Linux,macOS,Windows}_{Make,CMake}_{Debug,Release}_{serial,smpar,dmpar,dm_sm}
mv wats/work/output wats_Linux_CMake_Release_dmpar
# Repeat the above for each WRF variant to compare

# Optionally, download reference data to compare against
# 1. Go to https://dev.azure.com/WRF-CMake/wrf/_build?definitionId=5
# 2. Select a successful build from Branch "wrf-cmake"
# 3. Click on Summary
# 4. Download wats_Linux_Make_Debug_serial build artifact (~1 GB)
# 5. Extract archive to current folder

# Create plots
python wats/wats/plots.py compute wats_*
python wats/wats/plots.py plot --skip-detailed
ls wats/plots
# Compare magnitudes in nrmse.png and ext_boxplot.png with plots published in JOSS paper.
```

If you have any issues with the instructions above, please [open an issue](https://github.com/WRF-CMake/wats/issues/new).


## Copyright and license
Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.