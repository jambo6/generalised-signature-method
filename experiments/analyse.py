"""
analyse.py
===========================
Functions for analysing the output of the model runs. This performs analysis in the way it is described in the paper,
for other custom analysis, you will have to amend/build your own functions.
"""
from definitions import *
import numpy as np
from experiments.utils import create_run_frame
from src.visualization.critical_difference import plot_cd_diagram

transform_ids = {
    'depth_sig': ['depth', 'sig_tfm'],
    'non-learnt_augs': ['tfms'],
    'disintegrations_augs': ['tfms', 'disintegrations'],
    'learnt_augs': ['tfms', 'num_augments', 'augment_out'],
    'linear_learnt_augs': ['tfms', 'num_augments', 'augment_out'],
    'random_projections': ['tfms', 'num_projections', 'projection_channels'],
    'window': ['window'],
    'rescaling_and_norm': ['rescaling', 'normalisation']
}


def analyse(ex_dir, config_name, save_dir='analysis'):
    """Performs the analysis to generate the table maximised over the classifiers, and the cd-plot with ranks.

    Args:
        ex_dir (str): A directory containing savred experiment runs over a configuration.
        config_name (str): A key from the config dict in configurations.py.
        save_dir (str): The location to save the analysis. Set as 'analysis' to save in `ex_dir + '/' + analysis` else
            will save to `save_dir`.

    Returns:
        pd.DataFrame: The pivoted table.

    """
    assert config_name in transform_ids.keys(), "config name must be one of {}. Got {}. If this is a custom " \
                                                "configuration please add the transform_ids to the `transform_ids`" \
                                                "variable in `experiments.ingredients.analyse`."
    save_dir = save_dir if save_dir != 'analysis' else ex_dir + '/analysis'
    df = create_run_frame(ex_dir)

    df['transform_id'] = ''
    for col_name in transform_ids[config_name]:
        df['transform_id'] += col_name + '=' + df[col_name].astype(str)

    # Get results in ds_name x transform_id matrix
    results = df.groupby(['ds_name', 'transform_id', 'clf'])['acc.test'].apply(max)
    results = results.reset_index([1, 2]).drop('clf', axis=1)
    pivoted = results.pivot_table(values='acc.test', index='ds_name', columns='transform_id', aggfunc=np.max)
    # Plot and save the diagram
    plt, average_ranks = plot_cd_diagram(pivoted, return_items=True)

    # Save items
    save_pickle(pivoted, save_dir + '/results.pkl')
    save_pickle(average_ranks, save_dir + '/average_ranks.pkl')
    plt.savefig(save_dir + '/cd-diagram.png', dpi=300, bbox_inches='tight')

    return pivoted, average_ranks
