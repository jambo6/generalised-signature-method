"""
main.py
=========================
The main experiment run file.
"""
from definitions import *
from sacred import Experiment
import argparse
from experiments.dicts.configurations import configs
from experiments.dicts.data_dicts import datasets_dict
from experiments.ingredients.prepare.checkers import check_sklearn, check_learnt, check_meta
from experiments.ingredients.prepare.prepare_data import get_data, preprocess, compute_input_size
from experiments.utils import create_fso, basic_gridsearch, handle_error, set_completion_state
from experiments.ingredients.train import train_models
from experiments.ingredients.evaluate import evaluate_models

# For running in parallel
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--ex_name', help='The experiment name (determines the save folder).', default='copy_conf')
parser.add_argument('-c', '--config', help='The configuration entry key.', default=['test'], nargs='+')
parser.add_argument('-ds', '--datasets', help='The name of the datasets to run.', default=['ERing'], nargs='+')
parser.add_argument('-r', '--resume', help='Resume runs in a folder that already has saves.', action='store_true')
parser.add_argument('-j', '--jobs', help='Set True to parallelise the runs over the datasets.', default=1, type=int)
parser.add_argument('-s', '--save', help='Set True to save the model into a dill file.', action='store_true')
args = parser.parse_args()

# Handle the dataset arg
if len(args.datasets) == 1:
    if args.datasets[0] in datasets_dict.keys():
        args.datasets = datasets_dict[args.datasets[0]]

# Parallelise over the datasets if specified
if args.jobs > 1:
    datasets = ' '.join(args.datasets)
    config_strs = ' '.join(args.config)
    resume = '-r' if args.resume else ''
    save = '-s' if args.save else ''
    command = 'parallel -j {} --bar python main.py -c {{1}} -ds {{2}} {} {} ::: {} ::: {}' \
              ''.format(args.jobs, resume, save, config_strs, datasets)
    print('Running command: {}'.format(command))
    os.system(command)
    exit()
else:
    assert len(args.config) == 1, "Cannot have multiple configs when not in parallel mode. Set the -j flag to be > 1."

# Set the experiment and save folder
args.config = args.config[0]
ex_name = args.ex_name if args.ex_name != 'copy_conf' else args.config
save_dir = RESULTS_DIR + '/' + ex_name
ex = Experiment(ex_name)

# If the directory exists, proceed only when the user has confirmed they are aware of this
if os.path.exists(save_dir):
    if not args.resume:
        raise Exception("Runs already exist at: {}. \nPass the resume (-r) flag to confirm you are aware of this and "
                        "wish to proceed. \nElse delete the folder or change (-e) to a folder that doesn't yet exist."
                        .format(save_dir))

# Default configuration
@ex.config
def my_config():
    verbose = 2                     # Verbosity level
    gpu = True                      # Enable GPU
    sanity_dim = 1e5                # Max number of features
    ds_name = 'AtrialFibrillation'  # Dataset to load
    train_test_split = 'original'   # How to create train/test set
    scaling = 'stdsc'               # Feature scaling
    tfms = ['addtime']              # Basic augmentations
    rescaling = 'pre'               # Signature rescaling
    disintegrations = None          # Disintegrate into paths of size k
    num_augments = None             # Number of augmentations
    augment_out = None              # Number of output channels for each augmentation
    num_projections = None          # Number of projections
    projection_channels = None      # Number of channels for each projection
    normalisation = None            # Normalisation type
    window = ('Global', {})         # Windowing type and arguments
    sig_tfm = 'signature'           # Signature transform
    depth = 3                       # Signature depth
    clf = 'rf'                      # Classifier
    grid_search = False             # Whether to gridsearch over the parameters
    save_best_model = False         # Saves the best model as a .dill file.

# Main run file
@ex.main
def main(_run,
         ds_name,
         train_test_split,
         verbose,
         gpu,
         sanity_dim,
         scaling,
         tfms,
         clf,
         grid_search,
         rescaling,
         disintegrations,
         num_augments,
         augment_out,
         num_projections,
         projection_channels,
         window,
         depth,
         sig_tfm,
         normalisation,
         save_best_model
         ):
    try:
    # if True:
        # Add in save_dir
        _run.save_dir = '{}/{}'.format(save_dir, _run._id)

        ds_train, ds_test = get_data(ds_name, train_test_split)

        # Apply tfms here so they are not computed multiple times
        path_tfms, in_channels = preprocess(ds_train, scaling, tfms)

        # Open out some params
        ds_length, ds_dim, n_classes = ds_train.size(1), ds_train.size(2), ds_train.n_classes
        window_name, window_kwargs = window

        # Get in_channels with sanity check
        in_channels_clf, signature_channels = compute_input_size(
            in_channels, ds_length, window_name, window_kwargs, clf, disintegrations, augment_out, num_augments,
            num_projections, projection_channels, sig_tfm, depth, sanity_dim=sanity_dim
        )

        # Store some useful info to the saved metrics.
        _run.log_scalar('ds_length', ds_length)
        _run.log_scalar('ds_dim', ds_dim)
        _run.log_scalar('n_classes', n_classes)
        _run.log_scalar('n_train_samples', ds_train.size(0))
        _run.log_scalar('n_test_samples', ds_test.size(0))
        _run.log_scalar('in_channels_clf', in_channels_clf)

        # Perform checks to inform algorithm building
        is_learnt = check_learnt(num_augments, augment_out, normalisation)
        is_sklearn = check_sklearn(clf)
        is_meta = check_meta(window_name, clf)

        # Args used to build the signature model
        model_args = {
            'in_channels': in_channels,
            'signature_channels': signature_channels,
            'out_channels': n_classes if n_classes > 2 else 1,
            'ds_length': ds_length,
            'disintegrations': disintegrations,
            'num_augments': num_augments,
            'augment_out': augment_out,
            'num_projections': num_projections,
            'projection_channels': projection_channels,
            'window_name': window_name,
            'window_kwargs': window_kwargs,
            'sig_tfm': sig_tfm,
            'depth': depth,
            'rescaling': rescaling,
            'normalisation': normalisation,
            'clf': clf,
            'in_channels_clf': in_channels_clf,
            'gpu': gpu
        }

        # Train the small and large model.
        model_dict = train_models(
            _run, model_args, path_tfms, ds_train, is_learnt, is_sklearn, is_meta, grid_search, verbose=verbose
        )

        # Get training
        evaluate_models(_run, model_dict, ds_train, ds_test, is_sklearn, n_classes, save_best_model)

        # Note no errors
        _run.log_scalar('error', None)
        set_completion_state(_run, True)    # Mark as completed

    except Exception as e:
        handle_error(_run, e, print_error=True)


if __name__ == '__main__':
    # Configuration
    config = configs[str(args.config)]

    # Update the configuration with the CL-args.
    config['ds_name'] = args.datasets
    config['save_best_model'] = [args.save]

    # Create FSO (this creates a folder to log information into).
    create_fso(ex, save_dir, remove_folder=False)

    # Run a gridsearch over all parameter combinations.
    basic_gridsearch(ex, config, handle_completed_state=False)



