# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

from typing import Iterable
import os
from datetime import datetime, timedelta

import github

EXCLUDE = ['master']
KEEP_NEWEST = 2
REMOVE_AFTER = timedelta(days=1)
DRY_RUN = False

def get_date_utc(branch: github.Branch.Branch) -> datetime:
    return branch.commit.commit.author.date

g = github.Github(os.environ['GITHUB_TOKEN'])
repo = g.get_repo(os.environ['WATS_DATA_REPO'])
all_branches = repo.get_branches()
data_branches = filter(lambda branch: branch.name not in EXCLUDE, all_branches)
sorted_branches = sorted(data_branches, key=get_date_utc)

for i, branch in enumerate(sorted_branches):
    if i >= len(sorted_branches) - KEEP_NEWEST:
        print(f'Keeping {branch.name} (within newest {KEEP_NEWEST} branches)')
        continue
    age = datetime.utcnow() - get_date_utc(branch)
    if age < REMOVE_AFTER:
        print(f'Keeping {branch.name} (age={age} < {REMOVE_AFTER})')
        continue
    print(f'Deleting {branch.name} (age={age} >= {REMOVE_AFTER})')
    if not DRY_RUN:
        ref = repo.get_git_ref('heads/' + branch.name)
        ref.delete()
