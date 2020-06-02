"""
More specific functions for visualisation.
"""
from definitions import *
from src.visualization.general import *


def get_averaged_groupby(df, groupby, plot_column):
    """Gets the average of column after a groupby transformation.

    Example:
        We want the average accuracy at each signature depth over all datasets:
        >>> get_averaged_groupby(df, groupby='depth', plot_column='acc.mean')
        We can also specify some ybar errors with errors=column_name

    Args:
        df (pd.DataFrame): A DataFrame output from an experiment.
        groupby (str): The column with which to perform the groupby operation.
        plot_column (str): The column to plot the average of.
        kind (str): The type of plot ('line'/'bar')

    Returns:
        pd.Series: A series containing the averaged information..
    """
    return df.groupby(groupby)[plot_column].apply(np.mean)


def get_num_wins(df, column, metric):
    """Computes the number of times a given column achieves the highest metric score over the different datasets.

    Args:
        df (pd.DataFrame): A dataframe output from
        column (str): The column in the dataframe that we want to know the best scorer of.
        metric (str): A metric in the dataframe (column name).

    Returns:
        pd.Series: A series indexed by unique column with the number of times it won a dataset under the metric.
    """
    # Empty win count
    counts = pd.Series(index=df[column].unique(), data=0)

    # Find category winners
    winners = df.loc[df.groupby('ds_name')[metric].idxmax()][column].value_counts()

    # Update
    counts.update(winners)

    return counts


if __name__ == '__main__':
    from experiments.extractors import ExperimentToFrame
    df = ExperimentToFrame(ex_dir=MODELS_DIR + '/experiments/basic_order').generate_frame()
    df.drop('cv_params', axis=1)
    plot_averaged_groupby(df, 'tfms', 'acc.mean')
