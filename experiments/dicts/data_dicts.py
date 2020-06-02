"""
data_dicts.py
============================================
A collection of key, list pairs that correspond to a collection of grouped datasets. This is to make useful shortcuts
for running thins
"""
from definitions import *
import os

all_datasets = [x for x in os.listdir(DATA_DIR + '/processed') if not x.startswith('.')]

uea_datasets = ('ERing',
                'RacketSports',
                'PenDigits',
                'BasicMotions',
                'Libras',
                'JapaneseVowels',
                'AtrialFibrillation',
                'FingerMovements',
                'NATOPS',
                'Epilepsy',
                'LSST',
                'Handwriting',
                'UWaveGestureLibrary',
                'StandWalkJump',
                'HandMovementDirection',
                'ArticularyWordRecognition',
                'SelfRegulationSCP1',
                'CharacterTrajectories',
                'SelfRegulationSCP2',
                'Heartbeat',
                'FaceDetection',
                'SpokenArabicDigits',
                'EthanolConcentration',
                'Cricket',
                'DuckDuckGeese',
                'PEMS-SF',
                'InsectWingbeat',
                'PhonemeSpectra',
                'MotorImagery',
                'EigenWorms'
                )

datasets_dict = {
    'all': all_datasets,
    'uea': uea_datasets,
    'sc_ha': ['SpeechCommands', 'HumanActivity'],
}


assert all([x not in all_datasets for x in datasets_dict.keys()]), "Cannot have a key the same as an existing dataset."