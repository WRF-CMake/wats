#!/bin/bash

set -ex

# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    brew update
    # Install python 3.6.5 (https://stackoverflow.com/a/51125014) so that we can use netcdf wheels
    brew upgrade https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb

    python3 -V
    pip3 -V

fi
