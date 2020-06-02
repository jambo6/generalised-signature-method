""" Dictionaries of classifiers and parameter grids for testing """
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.models.nets import MLP, Linear, CNNResNet, GRU, MLPResNet


# Dictionary of classifiers, either sklearn or torch.nn.
CLASSIFIERS = {
    # Sklearn
    'lr': LogisticRegression(solver='lbfgs', multi_class='multinomial'),
    'rf': RandomForestClassifier(oob_score=True),

    # PyTorch
    'linear': Linear,
    'mlp': MLP,
    'mlpres': MLPResNet,
    'cnnres': CNNResNet,
    'gru': GRU,
}


# Parameter grids to search over if we are performing a gridsearch
GRIDSEARCH_PARAM_GRIDS = {
    'rf': {
        'classifier__n_estimators': [50, 100, 500, 1000],
        'classifier__max_depth': [2, 4, 6, 8, 12, 16, 24, 32, 45, 60, 80, None]
    },
}


# In the paper, we consider a small and large model for each classifier. The params for each are given below.
SML_CLASSIFIER_PARAM_GRIDS = {
    'lr': {
        'small': {
            'module__C': 0.01,
        },
        'large': {
            'module__C': 1,
        },
    },

    'rf': {
        'small': {
            'module__n_estimators': 100,
            'module__max_depth': 4,
        },
        'medium': {
            'module__n_estimators': 500,
            'module__max_depth': 10
        },
        'large': {
            'module__n_estimators': 1000,
            'module__max_depth': 50
        },

    },

    'mlp': {
        'small': {
            'module__hidden_channels': (128,),
            'module__normalization': True,
            'optim__lr': 0.1,
            'optim__max_epochs': 500,
            # 'optim__max_epochs': 1,
        },
        'large': {
            'module__hidden_channels': (512, 512, 512),
            'module__normalization': True,
            'optim__lr': 0.1,
            'optim__max_epochs': 1000,
            # 'optim__max_epochs': 1,
        },
    },

    'gru': {
        'small': {
            'module__hidden_channels': 32,
            'module__num_layers': 2,

            'optim__lr': 0.01,
            'optim__max_epochs': 500,
            # 'optim__max_epochs': 1,
        },
        'large': {
            'module__hidden_channels': 256,
            'module__num_layers': 3,

            'optim__lr': 0.01,
            'optim__max_epochs': 1000,
            # 'optim__max_epochs': 1,
        }
    },

    'mlpres': {
        'small': {
            'module__residual_channels': 32,
            'module__block_channels': 32,
            'module__num_blocks': 6,

            'optim__lr': 0.1,
            # 'optim__max_epochs': 500,
            'optim__max_epochs': 1,
        },
        'large': {
            'module__residual_channels': 128,
            'module__block_channels': 128,
            'module__num_blocks': 8,

            'optim__lr': 0.1,
            # 'optim__max_epochs': 1000,
            'optim__max_epochs': 1,
        }
    },

    'cnnres': {
        'small': {
            'module__residual_channels': 32,
            'module__block_channels': 32,
            'module__num_blocks': 6,
            'module__kernel_size': 4,
            'module__feedforward_channels': 256,

            'optim__lr': 0.001,
            'optim__max_epochs': 500,
            # 'optim__max_epochs': 1,
        },
        'large': {
            'module__residual_channels': 128,
            'module__block_channels': 128,
            'module__num_blocks': 8,
            'module__kernel_size': 8,
            'module__feedforward_channels': 1024,

            'optim__lr': 0.001,
            'optim__max_epochs': 1000,
            # 'optim__max_epochs': 1,
        }
    },

}


# Default set of params for all classifiers.
# This is needed due to limitations of GridSearchCV with neural nets. Requires initialisation and then hyperparams to
# be inserted. However, the nets cannot be initialised with no params, unlike sklearn models. So we give some default
# params that allow for initialisation and then they will be overwritten.
DEFAULT_CLF_PARAMS = {
    'linear': {
    },
    'mlp': {
        'hidden_channels': (5, 5),
        'normalization': [False],
    },
    'cnn': {
        'hidden_channels': (10, 10),
        'kernel_sizes': (3, 3),
        'mlp_hidden_channels': (10, 10)
    },
    'mlpres': {
        'residual_channels': 10,
        'block_channels': 10,
        'num_blocks': 2,
    },
    'cnnres': {
        'residual_channels': 10,
        'block_channels': 10,
        'kernel_size': 2,
        'num_blocks': 2,
        'feedforward_channels': 10,
    },
    'tcn': {
        'residual_channels': (10, 10),
        'block_channels': (10, 10),
        'kernel_size': 2,
        'num_blocks': 2,
        'feedforward_channels': 10,
    },
    'rnn': {
        'hidden_channels': 10,
        'num_layers': 2,
    },
    'gru': {
        'hidden_channels': 10,
        'num_layers': 2,
    },
    'lstm': {
        'hidden_channels': 10,
        'num_layers': 2,
    },
    'attention': {
        'residual_channels': 10,
        'hidden_channels': 10,
        'num_heads': 2,
        'num_layers': 2,
    },
    'dyadic': {
        'dyadic_depth': 3,
        'hidden_channels': 2
    }
}


ADDITIONAL_PARAM_GRIDS = {
    # Note that the min series length is 8
    # To edit how this works, or to add more options, edit the src.experiments.prepare_model.prepare_window function.
    # TODO Fix doing short time-series with num_windows.
    'window': {
        'Sliding': {
            'small': {
                'num_windows': 5,
            },
            'large': {
                'num_windows': 20
            }
        },
        'Expanding': {
            'small': {
                'num_windows': 5,
            },
            'large': {
                'num_windows': 20
            }
        }
    },

    'dyadic_meta': {
        'small': {
            'hidden_channels': 128,
        },
        'large': {
            'hidden_channels': 512
        }
    }
}

