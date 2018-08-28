# WRF-CMake Automated Testing Suite

The WRF-CMake Automated Testing Suite (WATS) compares WRF-/WPS-CMake and WRF-/WPS-Make NetCDF output files for several cases defined in namelist files (i.e. in *namelist.wps* and *namelist.input*).
WATS differs from the integration tests conducted in the Continuous Integration (CI) build matrix in that in WATS we check that WRF-/WPS-CMake do not produce different results to those produced using WRF-/WPS-Make.

## Copyright and license
Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.
