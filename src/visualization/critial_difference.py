"""
Functions used in plotting a critical difference diagram.

# Minor adaptation from: https://github.com/hfawaz/cd-diagram
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx

import warnings
warnings.simplefilter('ignore', UserWarning)

# Plot options
# matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def wilcoxon_holm(df, alpha=0.05):
    # Get a list of the scores
    classifiers = df.columns
    n_classifiers = len(df.columns)
    scores = df.values.T.tolist()
    # Run the Friedman test
    friedman_p_value = friedmanchisquare(*scores)[1]

    # Check null hypothesis
    if friedman_p_value >= alpha:
        print('NOTE: The null hypothesis over all classifiers cannot be rejected.')
        # raise Exception('The null hypothesis over all classifiers cannot be rejected.')

    # Get pairwise p values from the wilcoxon test
    pairwise = []
    for i in range(0, n_classifiers - 1):
        for j in range(i + 1, n_classifiers):
            # Remove missing values
            filter_ind=(~np.isnan(scores[i]))&(~np.isnan(scores[j]))
            scores_i=[k for indx,k in enumerate(scores[i]) if filter_ind[indx] == True]
            scores_j=[k for indx,k in enumerate(scores[j]) if filter_ind[indx] == True]

            p_value = wilcoxon(scores_i, scores_j, zero_method='pratt')[1]
            pairwise.append([classifiers[i], classifiers[j], p_value, False])

    # Sort ascending
    pairwise.sort(key=lambda x: x[2])

    # Apply Holms step down procedure
    k = len(pairwise)
    for i in range(k):
        holm_alpha = alpha / (k - i)
        if pairwise[i][2] <= holm_alpha:
            pairwise[i][3] = True
        else:
            break

    # Get ranks
    average_ranks = df.rank(axis=1, ascending=False).mean(axis=0).sort_values(ascending=False)

    return pairwise, average_ranks


def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """Plots a critical difference graph.

    Code taken from:  https://github.com/hfawaz/cd-diagram

    Draws a CD graph, which is used to display  the differences in methods performance.

    See Janez Demsar, Statistical Comparisons of Classifiers overMultiple Data Sets, 7(Jan):1--30, 2006.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)


    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    textsize=25

    cline += distanceh

    space_between_names = 0.3
    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.3, linesblank)
    height = cline + ((k + 1) / 2) * space_between_names + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=textsize)

    k = len(ssums)

    def filter_names(name):
        return name



    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=textsize)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=textsize)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.1
    side = -0.02
    height = 0.12

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def form_cliques(p_values, nnames):
    """Forms the cliques.

    Taken from: https://github.com/hfawaz/cd-diagram.
    """
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def plot_cd_diagram(df, alpha=0.05, return_items=False):
    """Plots the critical difference diagram given dataframe of classifiers and their accuracies on various datasets.

    Given a dataframe indexed by dataset name and columns being classifier name, with entries being the corresponding
    scores, plots a critical difference diagram of each of the classifiers.

    Args:
        df (pd.DataFrame): Dataframe with columns ['clf_name', 'ds_name', 'score'].
        alpha (float): Significance level
        return_items (bool): If true will return the plot and ranks

    Returns:
        plt, average_ranks (only if return_items=True)
    """
    pairwise, average_ranks = wilcoxon_holm(df, alpha=alpha)

    graph_ranks(average_ranks.values, average_ranks.keys(), pairwise, cd=None, reverse=True, width=9, textspace=1.5)

    if return_items:
        return plt, average_ranks



