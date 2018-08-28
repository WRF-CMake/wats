#!/bin/bash
set -ex

# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.
# Note: This script is used from the WRF/WPS repositories during CI.

cd $1

git init
git add .
git commit -m "automated commit"
git remote add origin git@github.com:$WATS_DATA_REPO.git
git push --force origin master:$TRAVIS_REPO_SLUG-$TRAVIS_BUILD_NUMBER-$MODE
