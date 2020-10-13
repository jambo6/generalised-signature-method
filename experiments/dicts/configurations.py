"""
config.py
========================
Model configurations.
"""

# Configuration files used in the paper
configs = {

    'test': {
        'depth': [3],
        'clf': ['rf'],
        'grid_search': [False],
    },

    'baseline': {
        'depth': [3],
        'clf': ['lr', 'rf', 'gru', 'cnnres'],
    },

    'depth_sig': {
        'depth': [1, 2, 3, 4, 5, 6],
        'sig_tfm': ['signature', 'logsignature'],
        'clf': ['lr', 'rf', 'gru', 'cnnres'],
    },

    'best_rf': {
        'depth': [1, 2, 3, 4, 5, 6],
        'sig_tfm': ['signature'],
        'tfms': [['addtime', 'basepoint']],
        'clf': ['rf'],
        'window': [
            ('Dyadic', {'depth': 2}),
            ('Dyadic', {'depth': 3}),
            ('Dyadic', {'depth': 4})
        ],
        'scaling': ['stdsc'],
        'rescaling': [None],
        'grid_search': [True]
    },

    'non-learnt_augs': {
        'tfms': [
            None,
            ['addtime'],
            ['basepoint'],
            ['penoff'],
            ['leadlag'],
            ['addtime', 'basepoint'],
            ['addtime', 'penoff'],
            ['leadlag', 'addtime'],
            ['leadlag', 'basepoint'],
            ['leadlag', 'penoff'],
            ['leadlag', 'addtime', 'penoff'],
            ['leadlag', 'addtime', 'basepoint'],
        ],
        'clf': ['lr', 'rf', 'gru', 'cnnres'],
    },

    'disintegrations_augs': {
        'tfms': [
            None,
            ['addtime'],
            ['addtime', 'basepoint'],
            ['addtime', 'penoff'],
            ['basepoint'],
            ['penoff'],
        ],
        'disintegrations': [1, 2, 3],
        'clf': ['lr', 'rf', 'gru', 'cnnres'],
    },

    'learnt_augs': {    # These are nonlinear
        'tfms': [
            ['addtime'],
            ['addtime', 'basepoint'],
            ['addtime', 'penoff'],
        ],
        'num_augments': [2, 5],
        'augment_out': [(16, 16, 3), (32, 32, 6)],
        'clf': ['gru', 'cnnres'],
    },

    'linear_learnt_augs': {
        'tfms': [
            None,
            ['addtime'],
            ['addtime', 'basepoint'],
            ['addtime', 'penoff'],
        ],
        'num_augments': [2, 5],
        'augment_out': [3, 6],
        'clf': ['gru', 'cnnres'],
    },

    'random_projections': {
        'tfms': [
            None,
            ['addtime'],
            ['addtime', 'basepoint'],
            ['addtime', 'penoff'],
            ['basepoint'],
            ['penoff'],
        ],
        'num_projections': [2, 5],
        'projection_channels': [3, 6],
        'clf': ['gru', 'cnnres'],
    },

    'window': {
        'window': [
            ('Global', {}),
            ('Sliding', {'size': 'small'}),
            ('Sliding', {'size': 'large'}),
            ('Expanding', {'size': 'small'}),
            ('Expanding', {'size': 'large'}),
            ('Dyadic', {'depth': 2}),
            ('Dyadic', {'depth': 3}),
            ('Dyadic', {'depth': 4}),
        ],
        'clf': ['lr', 'rf', 'gru', 'cnnres'],
    },

    'rescaling_and_norm': {
        'rescaling': [None, 'pre', 'post'],

        'normalisation': [
            None,
            {
                'norm_batch': True,
                'norm_graded': False,
                'norm_channel': False,
                'affine': False
            },
            {
                'norm_batch': True,
                'norm_graded': False,
                'norm_channel': True,
                'affine': False
            },
            {
                'norm_batch': True,
                'norm_graded': True,
                'norm_channel': True,
                'affine': False
            },
        ],
        'clf': ['gru', 'cnnres'],
    }
}


