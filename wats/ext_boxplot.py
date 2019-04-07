from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory

# Extended boxplots.
# See page 31 of http://biostat.mc.vanderbilt.edu/wiki/pub/Main/StatGraphCourse/graphscourse.pdf.
# This type of boxplot draws multiple boxes at different percentile pairs.
# There is no concept of whiskers or outliers.
# If the outer percentiles do not cover the whole range, i.e. are not 0 and 100,
# then the minimum and maximum values can be indicated.

def compute_extended_boxplot_stats(array, label=None, percentiles=[0.1, 1, 5, 25, 75, 95, 99, 99.9]) -> dict:
    ''' Alternative percentiles: [5, 12.5, 25, 37.5, 62.5, 75, 87.5, 95]
    '''
    array = np.asanyarray(array)
    p = np.percentile(array, percentiles)
    return dict(
        median=np.median(array),
        mean=np.mean(array),
        percentiles=p,
        min=np.min(array),
        max=np.max(array),
        values_min=array[array < p[0]],  # values between minimum and first given percentile
        values_max=array[array > p[-1]], # values between maximum and last given percentile
        label=label
        )

def plot_extended_boxplot(ax: Axes, boxplot_stats: List[dict],
                          positions: Optional[List[float]]=None,
                          showmeans: bool=True, showminmax: bool=True,
                          offscale_minmax: bool=False,
                          manage_xticks: bool=True) -> None:
    """
    Arguments like https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.bxp.html.

    offscale_minmax: Whether the min/max values should be labelled and drawn
                     at a fixed location independent of the y scale.
                     Useful when the min/max values are very different from the outer
                     percentiles.
    """
    N = len(boxplot_stats)

    if positions is None:
        positions = list(range(1, len(boxplot_stats) + 1))
    elif len(positions) != N:
        raise ValueError('List of boxplot statistics and positions values must have the same length')
    
    # TODO: this could be made a function of number of percentiles
    boxplot_facecolors = ['0.25', '0.50', '0.75', 'white']
    if any(len(stats['percentiles']) > 2*len(boxplot_facecolors) for stats in boxplot_stats):
        raise NotImplementedError('Too many percentiles')

    labels = []

    def get_boxplot_x0(boxplot_center, boxplot_width):
        boxplot_x0 = boxplot_center - boxplot_width / 2
        return boxplot_x0

    need_y_margin = False

    if offscale_minmax:
        buffer = 0.2
        percentile_min = min(s['percentiles'][0] for s in boxplot_stats)
        percentile_max = max(s['percentiles'][-1] for s in boxplot_stats)
        percentile_range = abs(percentile_max - percentile_min)
        thresh_min = percentile_min - percentile_range * buffer
        thresh_max = percentile_max + percentile_range * buffer

        tform = blended_transform_factory(ax.transData, ax.transAxes)
        text_dist = 0.1
        arrow_dist = 0.05

    for stats, position in zip(boxplot_stats, positions):
        labels.append(stats['label'] if stats['label'] is not None else position)

        percentiles = stats['percentiles']

        # Compute the number of iterations based on the number of percentiles
        percentiles_num_iter = int(len(percentiles)/2) # List of percentiles are always pairs so always a multiple of 2

        outermost_boxplot_width = 0.3
        boxplot_width = outermost_boxplot_width
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

        if showminmax:
            x0 = get_boxplot_x0(position, outermost_boxplot_width)
            x1 = x0 + outermost_boxplot_width

            # For the bottom
            if stats['median'] != percentiles[0]:
                if offscale_minmax and stats['min'] < thresh_min:
                    bottom_y = text_dist
                    ax.annotate('{:.3f}'.format(stats['min']), 
                        xy=(position, bottom_y), xycoords=tform,
                        xytext=(position, bottom_y - arrow_dist), textcoords=tform,
                        arrowprops=dict(arrowstyle='<-', facecolor='k', edgecolor='k'),
                        ha='center', va='top', color='k')
                    need_y_margin = True
                else:
                    bottom_y = stats['min']
                    ax.hlines(bottom_y, x0, x1, colors='k', linestyles='solid')

            # For the top
            if stats['median'] != percentiles[-1]:
                if offscale_minmax and stats['max'] > thresh_max:
                    top_y = 1 - text_dist
                    ax.annotate('{:.3f}'.format(stats['max']), 
                        xy=(position, top_y), xycoords=tform,
                        xytext=(position, top_y + arrow_dist), textcoords=tform,
                        arrowprops=dict(arrowstyle='<-', facecolor='k', edgecolor='k'),
                        ha='center', va='bottom', color='k')
                    need_y_margin = True
                else:
                    top_y = stats['max']
                    ax.hlines(top_y, x0, x1, colors='k', linestyles='solid')

    if need_y_margin:
        ax.set_ymargin(0.2)

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

    boxplot_stats = compute_extended_boxplot_stats(dist_norm)
    fig, ax = plt.subplots()
    plot_extended_boxplot(ax, [boxplot_stats])
    fig.savefig('ext_boxplot_test.png', dpi=200)
