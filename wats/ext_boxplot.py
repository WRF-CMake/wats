from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

def compute_extended_boxplot_stats(array, label=None, percentiles=[0.1, 1, 5, 25, 75, 95, 99, 99.9]) -> dict:
    array = np.asanyarray(array)
    p = np.percentile(array, percentiles)
    return {
        'total': array.size,
        'median': np.median(array),
        'mean': np.mean(array),
        'percentiles': p,
        'min': np.min(array),
        'max': np.max(array),
        'outliers_min': array[array < p[0]],
        'outliers_max': array[array > p[-1]],
        'label': label
    }

def plot_extended_boxplot(ax: Axes, boxplot_stats: List[dict],
                          positions: Optional[List[float]]=None,
                          showmeans: bool=True,
                          manage_xticks: bool=True) -> None:
    """
    Arguments like https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.bxp.html.
    """
    # Some code adapted from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/axes/_axes.py.

    N = len(boxplot_stats)

    if positions is None:
        positions = list(range(1, len(boxplot_stats) + 1))
    elif len(positions) != N:
        raise ValueError('List of boxplot statistics and positions values must have the same length')
    
    # TODO: this could be made a function of number of percentiles
    boxplot_facecolors = ['0.25', '0.50', '0.75', 'white']

    labels = []

    def get_boxplot_x0(boxplot_center, boxplot_width):
        boxplot_x0 = boxplot_center - boxplot_width / 2
        return boxplot_x0

    for stats, position in zip(boxplot_stats, positions):
        labels.append(stats['label'] if stats['label'] is not None else position)

        percentiles = stats['percentiles']

        #print('{}: min: {}, max: {}, mean: {}, median: {}, #outliers_min={}, #outliers_max={}, #total={}'.format(
        #    labels[-1], stats['min'], stats['max'], stats['mean'], stats['median'],
        #    len(stats['outliers_min']), len(stats['outliers_max']), stats['total']
        #    ))

        # Compute the number of iterations based on the number of percentiles
        percentiles_num_iter = int(len(percentiles)/2) # List of percentiles are always pairs so always a multiple of 2

        boxplot_width = 0.3
        for index in range(percentiles_num_iter):
            r = Rectangle((get_boxplot_x0(position, boxplot_width), percentiles[index]), 
                        boxplot_width, percentiles[-index-1] - percentiles[index], 
                        facecolor=boxplot_facecolors[index], fill=True, edgecolor='k')
            ax.add_patch(r)
            if index + 1 < percentiles_num_iter:
                boxplot_width = boxplot_width * 1.3

        x0 = get_boxplot_x0(position, boxplot_width)
        x1 = x0 + boxplot_width

        ax.hlines(stats['median'], x0, x1, colors='r', linestyles='solid')
        
        if showmeans:
            ax.plot(position, stats['mean'], marker='o', markersize=3, color="k")

        # Anything which is above the min and max range of percentiles,
        # we 'squash' into two lines which are half the percentile-range away
        # from the min and max percentiles.
        dist_outliers = (percentiles.max() - percentiles.min()) / 2

        blue = '#67a9cf'

        # For the bottom
        bottom_y = percentiles[0] - dist_outliers
        bottom_y_half = percentiles[0] - dist_outliers / 2
        ax.hlines(bottom_y, x0, x1, colors=blue, linestyles='solid')
        ax.vlines(position, bottom_y, percentiles[0], colors=blue, linestyles='dashed')
        ax.annotate('{:6.2f}'.format(stats['min']), 
                    xy=(position, bottom_y - abs(bottom_y)*0.01), ha='center', va='top', color=blue)
        ax.annotate('N={}'.format(len(stats['outliers_min'])),
                    xy=(position + 0.005, bottom_y_half), rotation=90, va='center', color=blue)

        # For the top
        top_y = percentiles[-1] + dist_outliers
        top_y_half = percentiles[-1] + dist_outliers / 2
        ax.hlines(top_y, x0, x1, colors=blue, linestyles='solid')
        ax.vlines(position, top_y, percentiles[-1], colors=blue, linestyles='dashed')
        ax.annotate('{:6.2f}'.format(stats['max']), 
                    xy=(position, top_y), ha='center', va='bottom', color=blue)
        ax.annotate('N={}'.format(len(stats['outliers_max'])),
                    xy=(position + 0.005, top_y_half), rotation=90, va='center', color=blue)

    if manage_xticks:
        ax.set_xmargin(0.05)

        axis = ax.xaxis

        locator = axis.get_major_locator()
        if not isinstance(axis.get_major_locator(), mticker.FixedLocator):
            locator = mticker.FixedLocator([])
            axis.set_major_locator(locator)
        locator.locs = np.array([*locator.locs, *positions])

        formatter = axis.get_major_formatter()
        if not isinstance(axis.get_major_formatter(), mticker.FixedFormatter):
            formatter = mticker.FixedFormatter([])
            axis.set_major_formatter(formatter)
        formatter.seq = [*formatter.seq, *labels]

        ax.autoscale_view()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dist_norm = np.random.normal(100, 30, 100000)

    #plt.boxplot(dist_norm)

    boxplot_stats = compute_extended_boxplot_stats(dist_norm)
    fig, ax = plt.subplots()
    plot_extended_boxplot(ax, [boxplot_stats])
    fig.savefig('foo.png', dpi=200)
