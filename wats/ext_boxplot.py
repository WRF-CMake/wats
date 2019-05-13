from typing import List, Optional

import math
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
                          positions: Optional[List[float]]=None, vert=True,
                          showmeans: bool=True, showminmax: bool=True,
                          offscale_minmax: bool=False, minmaxfmt='{:.1f}',
                          manage_ticks: bool=True) -> None:
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

    if vert:
        def doplot(*args, **kwargs):
            return ax.plot(*args, **kwargs)
        def doannotate(*args, **kwargs):
            return ax.annotate(*args, **kwargs)
        def dohlines(*args, **kwargs):
            return ax.hlines(*args, **kwargs)
        def dovlines(*args, **kwargs):
            return ax.vlines(*args, **kwargs)
        def doset_xmargin(*args, **kwargs):
            return ax.set_xmargin(*args, **kwargs)
        def doset_ymargin(*args, **kwargs):
            return ax.set_ymargin(*args, **kwargs)
    else:
        def doplot(*args, **kwargs):
            shuffled = []
            for i in range(0, len(args), 2):
                shuffled.extend([args[i + 1], args[i]])
            return ax.plot(*shuffled, **kwargs)

        def doannotate(*args, **kwargs):
            xy = kwargs.pop('xy')
            xy = (xy[1], xy[0])
            xytext = kwargs.pop('xytext', None)
            if xytext:
                xytext = (xytext[1], xytext[0])
            ha_ = kwargs.pop('ha')
            va_ = kwargs.pop('va')
            if ha_ == 'center' and va_ == 'top':
                ha = 'right'
                va = 'center_baseline'
            elif ha_ == 'center' and va_ == 'bottom':
                ha = 'left'
                va = 'center_baseline'
            else:
                raise NotImplementedError
            return ax.annotate(*args, xy=xy, xytext=xytext, ha=ha, va=va, **kwargs)

        dohlines = ax.vlines
        dovlines = ax.hlines
        doset_xmargin = ax.set_ymargin
        doset_ymargin = ax.set_xmargin

    if offscale_minmax:
        buffer = 0.2
        percentile_min = min(s['percentiles'][0] for s in boxplot_stats)
        percentile_max = max(s['percentiles'][-1] for s in boxplot_stats)
        percentile_range = abs(percentile_max - percentile_min)
        thresh_min = percentile_min - percentile_range * buffer
        thresh_max = percentile_max + percentile_range * buffer
        gap_start = 0.8
        gap_end = 0.85
        gap_min_start = percentile_min - percentile_range * (buffer * gap_start)
        gap_min_end = percentile_min - percentile_range * (buffer * gap_end)
        gap_max_start = percentile_max + percentile_range * (buffer * gap_start)
        gap_max_end = percentile_max + percentile_range * (buffer * gap_end)
        gap_linewidth = 0.5

    def get_boxplot_x0(boxplot_center, boxplot_width):
        boxplot_x0 = boxplot_center - boxplot_width / 2
        return boxplot_x0

    need_y_margin = False
    labels = []
    for stats, position in zip(boxplot_stats, positions):
        labels.append(stats['label'] if stats['label'] is not None else position)

        percentiles = stats['percentiles']

        # Compute the number of iterations based on the number of percentiles
        percentiles_num_iter = int(len(percentiles)/2) # List of percentiles are always pairs so always a multiple of 2

        outermost_boxplot_width = 0.3
        boxplot_width = outermost_boxplot_width
        for index in range(percentiles_num_iter):
            x = get_boxplot_x0(position, boxplot_width)
            y = percentiles[index]
            width = boxplot_width
            height = percentiles[-index-1] - percentiles[index]
            if not vert:
                width, height = height, width
                x, y = y, x
            r = Rectangle((x,y), width, height,
                          facecolor=boxplot_facecolors[index], fill=True, edgecolor='k')
            ax.add_patch(r)
            if index + 1 < percentiles_num_iter:
                boxplot_width = boxplot_width * 1.3

        x0 = get_boxplot_x0(position, boxplot_width)
        x1 = x0 + boxplot_width

        dohlines(stats['median'], x0, x1, colors='r', linestyles='solid')
        
        if showmeans:
            doplot(position, stats['mean'], marker='o', markersize=3, color="k")

        if showminmax:
            x0 = get_boxplot_x0(position, outermost_boxplot_width)
            x1 = x0 + outermost_boxplot_width

            # For the bottom
            if stats['median'] != percentiles[0]:
                if offscale_minmax and stats['min'] < thresh_min:
                    dovlines(position, percentiles[0], gap_min_start, colors='k', linestyles='solid')
                    dovlines(position, gap_min_end, thresh_min, colors='k', linestyles='solid')
                    dy = 0.3 * (gap_min_start - gap_min_end)
                    doplot([x0, x1], [gap_min_start - dy, gap_min_start + dy], color='k', lw=gap_linewidth)
                    doplot([x0, x1], [gap_min_end - dy, gap_min_end + dy], color='k', lw=gap_linewidth)
                    text_buffer = 2 * (gap_min_start - gap_min_end)
                    doannotate(minmaxfmt.format(stats['min']), 
                        xy=(position, thresh_min - text_buffer),
                        ha='center', va='top', color='k')
                    need_y_margin = True
                else:
                    dovlines(position, percentiles[0], stats['min'], colors='k', linestyles='solid')

            # For the top
            if stats['median'] != percentiles[-1]:
                if offscale_minmax and stats['max'] > thresh_max:
                    dovlines(position, percentiles[-1], gap_max_start, colors='k', linestyles='solid')
                    dovlines(position, gap_max_end, thresh_max, colors='k', linestyles='solid')
                    dy = 0.3 * (gap_max_end - gap_max_start)
                    doplot([x0, x1], [gap_max_start - dy, gap_max_start + dy], color='k', lw=gap_linewidth)
                    doplot([x0, x1], [gap_max_end - dy, gap_max_end + dy], color='k', lw=gap_linewidth)
                    text_buffer = 2 * (gap_max_end - gap_max_start)
                    doannotate(minmaxfmt.format(stats['max']), 
                        xy=(position, thresh_max + text_buffer),
                        ha='center', va='bottom', color='k')
                    need_y_margin = True
                else: 
                    dovlines(position, percentiles[-1], stats['max'], colors='k', linestyles='solid')

    if need_y_margin:
        doset_ymargin(0.1)

    if manage_ticks:
        doset_xmargin(0.05)

        axis_name = "x" if vert else "y"
        axis = getattr(ax, f"{axis_name}axis")

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
