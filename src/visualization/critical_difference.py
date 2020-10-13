"""
Functions used in plotting a critical difference diagram.

# Adaptation from https://github.com/biolab/orange3/blob/master/Orange/evaluation/scoring.py
"""

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from definitions import *
import math
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', UserWarning)

# Plot options
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def wilcoxon_holm(df, alpha=0.05,refmethod=None):
    # order the dataframe by average rank
    average_ranks = df.rank(axis=1, ascending=False).mean(axis=0).sort_values(ascending=True)
    # Get a list of classifiers
    classifiers = average_ranks.index
    # Get a list of the scores
    scores = df[classifiers].values.T.tolist()
    n_classifiers = len(df.columns)
    # Run the Friedman test
    friedman_p_value = friedmanchisquare(*scores)[1]

    # Check null hypothesis
    if friedman_p_value >= alpha:
        print('NOTE: The null hypothesis over all classifiers cannot be rejected.')

    # Get pairwise p values from the wilcoxon test by comparing highest rank vs the others recursively
    pairwise = []
    for i in range(n_classifiers-1):
        # Compare to classifiers of lower rank
        j = i+1
        # Max number of comparison in this clique is n_classifiers-i-1, use bonferonni correction
        holm_alpha = alpha / (n_classifiers - i-1)
        p_value = 1.0
        # Compare until one pair is significant
        while (p_value >= holm_alpha) & (j < n_classifiers):
            # Handle NA values by removing them for each test
            filter_ind = (~np.isnan(scores[i])) & (~np.isnan(scores[j]))
            scores_i = [k for indx, k in enumerate(scores[i]) if filter_ind[indx] == True]
            scores_j = [k for indx, k in enumerate(scores[j]) if filter_ind[indx] == True]
            p_value = wilcoxon(scores_i, scores_j, zero_method='wilcox', alternative='greater')[1]
            if p_value >= holm_alpha:
                pairwise.append([classifiers[i], classifiers[j], p_value, False])
            else:
                pairwise.append([classifiers[i], classifiers[j], p_value, True])
            j += 1
    return pairwise, average_ranks


def graph_ranks(avranks, names,pvalues, lowv=None, highv=None,
                width=6, textspace=1, reverse=True, filename=None, textfontsize=12, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        pvalues (list): pvalues and significance boolean of pairs of names
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        textfontsize (int,optional): method names font size
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth element in a list.
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

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    dict_pvalues={}
    for pvalue in pvalues:
        dict_pvalues[(pvalue[0],pvalue[1])]=[pvalue[2],pvalue[3]]
        dict_pvalues[(pvalue[1], pvalue[0])] = [pvalue[2], pvalue[3]]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    # get pairs of non significant methods

    def no_longer(ij_tuple, notSig):
        i, j = ij_tuple
        for i1, j1 in notSig:
            if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                return False
        return True

    def get_lines(names, dict_pvalues):
        # remove not significant
        notSig = [(names.index(i), names.index(j)) for i, j in dict_pvalues.keys()
                  if not dict_pvalues[(i, j)][1]]
        # keep only longest
        longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]
        return longest

    lines = get_lines(nnames,dict_pvalues)
    linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

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

    textfontsize = textfontsize

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    # Draw main ranks line
    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

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
             ha="center", va="bottom",fontsize=textfontsize)

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center", fontsize=textfontsize)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center", fontsize=textfontsize)

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv)
    else:
        begin, end = rankpos(highv), rankpos(highv)

    line([(begin, distanceh), (end, distanceh)], linewidth=0.7)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2
        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=2.5)
            start += height

    draw_lines(lines)

    if filename:
        print_figure(fig, filename, **kwargs)


def plot_cd_diagram(df, alpha=0.05, width=8, textspace=1.5,textfontsize=12, return_items=False):
    # Compute ranks
    pairwise, average_ranks = wilcoxon_holm(df,alpha=alpha)
    graph_ranks(average_ranks.values, average_ranks.keys(), pairwise, width=width, textspace=textspace,
                textfontsize=textfontsize)
    if return_items:
        return plt, average_ranks


if __name__ == '__main__':
    df_test = pd.DataFrame({'acc.test': np.random.random(15), 'ds_name': np.repeat(['a', 'b', 'c', 'd', 'e'], 3),
                          'tfms':np.tile(['algo_1', 'algo_2', 'algo_3'], 5)})
    df_analysis = df_test.pivot_table(values='acc.test', index=['ds_name'], columns=['tfms'], aggfunc=np.max)
    plot_cd_diagram(df_analysis)
    plt.show()