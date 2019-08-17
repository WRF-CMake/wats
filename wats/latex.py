from typing import List
import os
import numpy as np
import pandas as pd

def abs_err_to_latex(configs: List[str], quantities: List[str], means, stds, maxs) -> str:
    if all('CMake' for cfg in configs):
        configs = [cfg.replace('CMake/', '') for cfg in configs]
    index = pd.MultiIndex.from_product([configs, ['Mean', 'SD', 'Max']],
                                       names=['Configuration', 'Statistic'])
    data = np.empty([len(configs) * 3, len(quantities)], float)
    data[0::3] = means
    data[1::3] = stds
    data[2::3] = maxs
    df = pd.DataFrame(data, index=index, columns=quantities)
    latex = df.to_latex(
        escape=False,  # columns labels contain tex code
        longtable=True,
        float_format='{:0.1e}'.format)
    latex = latex.replace('dm_sm', 'dm\_sm')
    return latex


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'plots_make_cmake_d02_t6')
    configs = open(os.path.join(path, 'trial_labels.csv')).read().split(',')
    quantity_labels = open(os.path.join(path, 'quantity_labels.csv')).read().split(',')
    quantity_units = open(os.path.join(path, 'quantity_units.csv')).read().split(',')
    quantities = [f'{l} in {u}' for l,u in zip(quantity_labels, quantity_units)]
    means = np.loadtxt(os.path.join(path, 'mae.csv'))
    stds = np.loadtxt(os.path.join(path, 'ae_std.csv'))
    maxs = np.loadtxt(os.path.join(path, 'ae_max.csv'))
    l = abs_err_to_latex(configs, quantities, means, stds, maxs)
    print(l)
