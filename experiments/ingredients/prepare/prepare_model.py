"""
model.py
==========================
Functions that take a dictionary of strings as an input, and return an initialised model class.
"""
import inspect
import torch
from torch import nn, optim
from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, EpochScoring
from skorch.dataset import CVSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from experiments.ingredients.prepare.checkers import check_sklearn
from src.omni.base import NullTransformer
from src.features.signatures.compute import ComputeWindowSignature, SklearnComputeWindowSignature
from src.features.signatures.transforms import LearntAugment, NullAugmenter, Disintegrator, RandomProjections, LearntProjections
from experiments.dicts.grids import CLASSIFIERS, DEFAULT_CLF_PARAMS
from src.features.signatures.functions import push_batch_trick, combine_tricks
from src.models.normalization import GradedNormalization
from src.models.utils import Sklearnify, TrickSignature
from src.models.nets import DyadicModelIndividual
from experiments.utils import get_freest_gpu

device = torch.device('cuda:{}'.format(get_freest_gpu())) if torch.cuda.is_available() else torch.device('cpu')


class SignatureStringModel(nn.Module, BaseEstimator, TransformerMixin):
    """Builds a signature model from strings that are passed to the experiment configuration.

    This string format is necessary if we wish to perform hyperparameter search in parallel as the class must be
    pickleable. This is built for the experiment runs and so takes args analogous to the model_args input.
    """
    def __init__(self,
                 in_channels=None,
                 signature_channels=None,
                 out_channels=None,
                 ds_length=None,
                 disintegrations=None,
                 num_augments=None,
                 augment_out=None,
                 num_projections=None,
                 projection_channels=None,
                 window_name=None,
                 window_kwargs=None,
                 sig_tfm=None,
                 depth=None,
                 rescaling=None,
                 normalisation=None,
                 clf=None,
                 in_channels_clf=None,
                 gpu=None,
                 is_meta=None,
                 **dunder_kwargs):
        super(SignatureStringModel, self).__init__()

        # Make empty clf params (updated in _init_dunder_kwargs)
        self.clf_params = {}

        # Sort augmentations
        self.num_augments, self.augment_out = 1, 0
        if isinstance(num_augments, int) and (isinstance(augment_out, int) or isinstance(augment_out, tuple)):
            self.num_augments, self.augment_out = num_augments, augment_out

        # Sort random_projections
        self.num_projections, self.projection_channels = 0, 0
        if all([isinstance(num_projections, int), isinstance(projection_channels, int)]):
            self.num_projections, self.projection_channels = num_projections, projection_channels

        # Init params
        self.in_channels = in_channels
        self.signature_channels = signature_channels
        self.out_channels = out_channels
        self.ds_length = ds_length
        self.disintegrations = disintegrations
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.rescaling = rescaling
        self.clf = clf
        self.normalisation_args = normalisation
        self.in_channels_clf = in_channels_clf
        self.gpu = gpu
        self.is_meta = is_meta

        if self.normalisation_args is None:
            self.normalisation = None

        # Update hyperparameters
        if isinstance(dunder_kwargs, dict):
            self._init_dunder_kwargs(dunder_kwargs)

        # Init model
        self._init_learnt_augment()
        self._init_random_projections()
        self._init_normalisation()
        self._init_signatures()
        self._init_classifier()

    def _init_dunder_kwargs(self, dunder_kwargs):
        """Sets hyperparameters ready for model building.

        Must input parameters of the following form for networks:
            params = {
                'arg__param_1': value_1,
                'arch__param_1': value_2,
            }
        That is, parameters keys must follow one of ['arg', 'arch', 'optim'].
            'arg' -  For learnt augmentations, either 'num_augments' or 'augment_dim' should follow it.
            'arch' - For the architecture. Any params are passed the the model classifier.

        NOTE: If you are using an sklearn classifier then all params should follow the 'arch' keyword.

        Args:
            params (dict): A dictionary of parameters that must be specified as stated above.

        Returns:
            self
        """
        for key, value in dunder_kwargs.items():
            type, name = key.split('__')
            if type == 'aug':
                # Augments
                if name == 'num_augments':
                    self.num_augments = value if isinstance(value, int) else 1
                elif name == 'augment_out':
                    self.augment_out = value if isinstance(value, int) else 0

                # Projections
                if name == 'num_projections':
                    self.num_projections = value if isinstance(value, int) else 1
                elif name == 'projection_channels':
                    self.projection_channels = value if isinstance(value, int) else 0
            elif type == 'arch':
                self.clf_params[name] = value
            else:
                raise ValueError("{} not a valid type. Must be one of ['aug', 'arch'].".format(key))
        return self

    def _init_learnt_augment(self):
        self.learnt_tfms = NullAugmenter()
        if isinstance(self.augment_out, int):
            if self.augment_out > 0:
                self.learnt_tfms = LearntProjections(
                    in_channels=self.in_channels,
                    num_projections=self.num_augments,
                    projection_channels=self.augment_out
                )
        elif isinstance(self.augment_out, tuple):
            self.learnt_tfms = LearntAugment(
                in_channels=self.in_channels,
                num_augments=self.num_augments,
                augment_sizes=self.augment_out,
                keep_original=True
            )

    def _init_random_projections(self):
        self.random_projections = NullAugmenter()
        if self.num_projections > 0:
            self.random_projections = RandomProjections(
                self.in_channels, self.num_projections, self.projection_channels
            )

    def _init_normalisation(self):
        if self.normalisation_args is not None:
            self.normalisation = GradedNormalization(
                (None, self.signature_channels), self.in_channels, self.depth, **self.normalisation_args
            )

    def _init_signatures(self):
        # Window signature
        self.window = ComputeWindowSignature(
            self.window_name, self.window_kwargs, self.ds_length, self.sig_tfm, self.depth, self.rescaling,
            normalisation=self.normalisation
        )
        self.num_windows = self.window.num_windows(self.ds_length)

    def _init_classifier(self):
        # Get the model
        classifier = CLASSIFIERS[self.clf]

        # Update with defaults if the arguments have not been specified
        clf_args = inspect.getfullargspec(classifier).args
        if 'length' in clf_args:
            self.clf_params['length'] = self.window.num_windows(self.ds_length)
        current_args = list(self.clf_params.keys()) + ['self', 'in_channels', 'out_channels']
        for arg in DEFAULT_CLF_PARAMS[self.clf].keys():
            if arg not in current_args:
                self.clf_params[arg] = DEFAULT_CLF_PARAMS[self.clf][arg]

        self.classifier = classifier(in_channels=self.in_channels_clf, out_channels=self.out_channels, **self.clf_params)

        # Update if Dyadic
        if self.is_meta is True:
            self.classifier = DyadicModelIndividual(
                out_channels=self.out_channels, dyadic_depth=self.window_kwargs['depth'], hidden_channels=100,
                model=self.classifier
            )

    def disintegrate(self, path):
        """ Perform disintegrations. """
        disintegrator = Disintegrator(size=self.disintegrations)
        path_disint = disintegrator.transform(path) if self.disintegrations is not None else path
        return push_batch_trick(path_disint)

    def forward(self, path):
        # Disintegrations
        trick_info_disintegration, disintegrated_path = self.disintegrate(path)

        # Any learned augmentations, batch trick is used if we have multiple learnt augmentations
        transformed_path = self.learnt_tfms(disintegrated_path)

        # Random projection
        transformed_path = self.random_projections(transformed_path)

        # Trick for if multiple augmentations used
        trick_info_transform, tricked_path = push_batch_trick(transformed_path)

        # Combine tricks
        trick_info = combine_tricks(trick_info_disintegration, trick_info_transform)

        # Compute signatures over windows
        signatures = self.window(tricked_path, path.size(-1), trick_info)

        # Apply the model
        result = self.classifier(signatures)

        # No final non-linearity, we'll include that as part of the loss function

        return result


def prepare_learnt_model(model_args, path_tfms, is_meta, verbose=2):
    """Model builder if learnt transforms are involved.

    The key difference (as explained in prepare_non_learnt_model) between this function and prepare_non_learnt_model
    is that the

    Args:
        model_args (dict): Experiment model args as defined in the main experiment function.
        path_tfms (Pipeline): An sklearn pipeline of path transformations to be applied before model training.
        is_meta (bool): Set True for a dyadic meta model.
        verbose (int): Output verbosity level.

    Returns:

    """
    # Initialise the signature string class.
    model_args['is_meta'] = is_meta
    module = SignatureStringModel(**model_args)

    model = NeuralNetClassifier(
        module=module,
        criterion=nn.BCEWithLogitsLoss if model_args['out_channels'] == 1 else nn.CrossEntropyLoss,
        batch_size=64,
        verbose=verbose,
        iterator_train__drop_last=True,
        callbacks=[
            ('scheduler', LRScheduler(policy='ReduceLROnPlateau')),
            ('val_stopping', EarlyStopping(monitor='valid_loss', patience=30)),
            ('checkpoint', CustomCheckpoint(monitor='valid_loss_best')),
            ('scorer', EpochScoring(custom_scorer, lower_is_better=False, name='true_acc'))
        ],
        train_split=CVSplit(cv=5, random_state=1, stratified=True),
        device=device if model_args['gpu'] else 'cpu',
    )
    pipeline = Pipeline([
        *path_tfms,
        ('classifier', model)
    ])
    return pipeline


def prepare_non_learnt_model(model_args, path_tfms, is_meta=False, verbose=2):
    """Prepares a model that does not use learnt projections.

    We have a separate function for learnt and non-learnt transforms since the learnt transform requires signatures to
    be computed each time we evaluate over a batch. If the model is non-learnt then all signatures can be computed in
    advance which significantly reduces the training time.

    Args:
        model_args (dict): Experiment model arguments. See the main e
        path_tfms (Pipeline): A pipeline of path transformations.
        is_meta (bool): Set true if the model is a dyadic-meta-model.
        verbose (int): Level of output verbosity.

    Returns:
        sklearn transformer: Trained model.
    """
    # Check if sklearn
    is_sklearn = check_sklearn(model_args['clf'])

    # Window arguments
    arg_names = ['ds_length', 'window_name', 'window_kwargs', 'depth', 'sig_tfm', 'rescaling']
    window_args = {
        k: v for k, v in model_args.items() if k in arg_names
    }

    # Setup the disintegrator
    darg = model_args['disintegrations']
    disintegrator = Disintegrator(size=darg) if isinstance(darg, int) else NullTransformer()

    # Random projector
    proj_args = model_args['num_projections'], model_args['projection_channels']
    proj_bool = all([x is not None for x in proj_args])
    if proj_bool:
        projector = Sklearnify(RandomProjections(model_args['in_channels'], proj_args[0], proj_args[1]))
    else:
        projector = NullTransformer()

    # Signature computation
    window = SklearnComputeWindowSignature(**window_args) if is_sklearn else TrickSignature(ComputeWindowSignature(**window_args))

    # Set classifier
    classifier = CLASSIFIERS[model_args['clf']]

    if not is_sklearn:
        # We initialise with some defaults if not specified.
        init_clf_params = DEFAULT_CLF_PARAMS[model_args['clf']]
        clf_args = inspect.getfullargspec(classifier).args
        if 'length' in clf_args:
            init_clf_params['length'] = window.module.num_windows(model_args['ds_length'])

        # Update if Dyadic
        module = classifier(in_channels=model_args['in_channels_clf'], out_channels=model_args['out_channels'],
                            **init_clf_params)
        if is_meta:
            module = DyadicModelIndividual(
                out_channels=model_args['out_channels'], dyadic_depth=window_args['depth'], hidden_channels=100,
                model=module
            )

        classifier = NeuralNetClassifier(
            module=module,
            criterion=nn.BCEWithLogitsLoss if model_args['out_channels'] == 1 else nn.CrossEntropyLoss,
            batch_size=64,
            verbose=verbose,
            optimizer=optim.Adam,
            iterator_train__drop_last=True,
            callbacks=[
                ('scheduler', LRScheduler(policy='ReduceLROnPlateau')),
                ('val_stopping', EarlyStopping(monitor='valid_loss', patience=30)),
                ('checkpoint', CustomCheckpoint(monitor='valid_loss_best')),
                ('scorer', EpochScoring(custom_scorer, lower_is_better=False, name='true_acc'))
            ],
            train_split=CVSplit(cv=5, random_state=1, stratified=True),
            device=device if model_args['gpu'] else 'cpu',
        )

    model = Pipeline([
        *path_tfms,
        ('disintegrator', disintegrator),
        ('random_projections', projector),
        ('signatures', window),
        ('classifier', classifier)
    ])

    return model


class CustomCheckpoint(Checkpoint):
    """ Custom checkpoint that saves the model to itself rather than a file. """
    def save_model(self, net):
        self.state_dict = net.module.state_dict()


def custom_scorer(net, ds, y=None):
    """ Custom scoring function that works for binary problems. Skorch scorer does not work correctly for binary. """
    output = net.predict_proba(ds)

    # Make predictions
    if output.shape[1] > 1:
        probas = torch.softmax(torch.Tensor(output), dim=1)
        preds = probas.argmax(dim=1)
    else:
        probas = torch.sigmoid(torch.Tensor(output))
        preds = torch.round(probas)

    score = accuracy_score(preds, y)

    return score


