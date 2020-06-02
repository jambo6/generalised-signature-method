"""
utils.py
================================================
Various method used in setting up, running, and extracting sacred experiments.
"""
from definitions import *
import os, shutil
from pprint import pprint
import numpy as np
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import logging


def set_ex_logger(ex, level='WARNING'):
    """ Sets the experiment logging level. """
    logger = logging.getLogger('logger')
    logger.setLevel(getattr(logging, level))
    ex.logger = logger


def create_fso(ex, directory, remove_folder=True):
    """
    Creates a file storage observer for a given experiment in the specified directory.

    Check sacred docs for a full explanation but this just sets up the folder to save the information from the runs of
    the given experiment.

    NOTE: This is currently setup to delete/replace the specified folder if it already exists. This should be changed if
    it is not the desired behaviour.

    Args:
        ex (sacred.Experiment): The sacred experiment to be tracked.
        directory (str): The directory to 'watch' the experiment and save information to.

    Returns:
        None
    """
    if remove_folder:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
    ex.observers.append(FileStorageObserver(directory))


def ready_mongo_observer(ex, db_name='sacred', url='localhost:27017'):
    """Readies a mongo observer for use with sacred.

    Args:
        ex (sacred.Experiment): Sacred experiment to track.
        db_name (str): Name of the mongo database.
        url (str): Host location.
    """
    ex.observers.append(MongoObserver(url=url, db_name=db_name))


def basic_gridsearch(ex, grid, verbose=2, handle_completed_state=True):
    """Basic gridsearch for a sacred experiment.

    Given an experiment and a parameter grid, this will iterate over all possible combinations of parameters specified
    in the grid. In an iteration, the experiment configuration is updated and the experiment is run.

    Args:
        ex (sacred.Experiment): A sacred experiment.
        grid (dict, list): Parameter grid, analogous setup to as with sklearn gridsearches. If a list is specified then
            it will assume it is from a restart and continue as normal.
        verbose (int): Output verbosity level.
        handle_completed_state (bool): Set True to examine whether the parameters have already been run and additionally
            if that run was marked as completed. NOTE: You need to be careful with this option. It will only work if the
            completion state is being set at the end of an experiment run. If it is not being set then it will always
            delete and rerun.

    Returns:
        None
    """
    # Setup the grid
    if isinstance(grid, dict):
        param_grid = list(ParameterGrid(grid))
    else:
        param_grid = grid
    grid_len = len(param_grid)

    for i, params in tqdm(enumerate(param_grid)):
        # Print info
        if verbose > 0:
            print('\n\n\nCONFIGURATION {} of {}\n'.format(i + 1, grid_len) + '-' * 100)
            pprint(params)
            print('-' * 100)

        # Skip if done
        if handle_completed_state:
            if check_run_existence(ex, params):
                continue

        # Update configuration and run
        ex.run(config_updates=params, info={})


def check_run_existence(ex, params):
    """ Checks if the parameters for a given run already exist. """
    save_dir = ex.observers[0].basedir
    completed_bool = False

    # If directory exists, load in all configurations
    if os.path.exists(save_dir):
        configs, run_nums = load_configs(save_dir)
    else:
        return False

    # Remove additional keys that are made by sacred
    for i in range(len(configs)):
        configs[i] = {k: v for k, v in configs[i].items() if k in params.keys()}

    # If the model was completed, then do not run it again. If it was not completed then remove it and start afresh.
    for config, run_num in zip(configs, run_nums):
        loc = save_dir + '/' + run_num + '/other/completion_state.pkl'
        if not os.path.exists(loc):
            return False
        is_completed = load_pickle(loc)

        # Skip if completed, else if same params and not completed then remove and run again
        if configs == params:
            if is_completed:
                completed_bool = True
                print('Skipping {} as already done.'.format(params))

            if not is_completed:
                os.rmdir(save_dir + '/' + run_num)

    return completed_bool


def set_completion_state(_run, state):
    """ Sets the completion state as True or False for help with reruns. """
    assert any([state == True, state == False])
    save_pickle(state, _run.save_dir + '/other/completion_state.pkl')


def handle_error(_run, e, print_error=True, err_name='error'):
    """ Saves error to run, writes it to a file and prints to the console. """
    # Save error information
    _run.log_scalar(err_name, str(e))
    with open(_run.save_dir + '/{}.txt'.format(err_name), 'w') as f:
        f.write(str(e))

    # Print error to console
    if print_error:
        print('\nERROR\n' + '~' * 30 + '\n' + 'MESSAGE: {}'.format(str(e)))


def get_freest_gpu():
    """ GPU with most available memory. """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def get_run_nums(ex_dir):
    """ Extracts the run folder names from an experiment directory. """
    not_in = ['_sources', '.ipynb_checkpoints', 'checkpoint', 'analysis']
    run_nums = [x for x in os.listdir(ex_dir) if x not in not_in]
    return run_nums


def extract_config(loc):
    """ Extracts the configuration from the directory. """
    config = load_json(loc + '/config.json')
    del config['seed']
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(loc + '/metrics.json')

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def load_configs(ex_dir):
    """ Loads all configuration files into a list from the given experiment directory. """
    configs = []
    run_nums = get_run_nums(ex_dir)
    for run_num in run_nums:
        loc = ex_dir + '/' + run_num
        try:
            configs.append(extract_config(loc))
        except:
            raise Exception("Cannot load config in {}. Please remove this from the directory to proceed".format(loc))
    return configs, run_nums


def create_run_frame(ex_dir):
    """Creates a dataframe from the run saves.

    Args:
        ex_dir (str): The experiment directory.

    Returns:
        pd.DataFrame: A pandas dataframe containing all results from the run.
    """
    run_nums = get_run_nums(ex_dir)

    frames = []
    for run_num in run_nums:
        loc = ex_dir + '/' + run_num
        try:
            config = extract_config(loc)
        except Exception as e:
            print('Could not load config at: {}. Failed with error:\n\t"{}"'.format(loc, e))
        try:
            metrics = extract_metrics(loc)
        except Exception as e:
            print('Could not load metrics at: {}. Failed with error:\n\t"{}"'.format(loc, e))

        # Create a config and metrics frame and concat them
        config = {str(k): str(v) for k, v in config.items()}    # Some dicts break for some reason
        df_config = pd.DataFrame.from_dict(config, orient='index').T
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df = pd.concat([df_config, df_metrics], axis=1)
        df.index = [int(run_num)]
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)

    # Reorder some cols
    cols_front = ['ds_name']
    df = df[cols_front + [x for x in df.columns if x not in cols_front]]

    # Make numeric cols when possible
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'), axis=1)

    return df




