"""
Gets summary information for the datasets and converts to a use-able format.
"""
from definitions import *
import pandas as pd
import numpy as np


def clean_uv(df):
    df.rename(columns={'Test ': 'Test', 'Train ': 'Train'}, inplace=True)
    df.drop(['ID', 'Data donor/editor'], axis=1, inplace=True)
    return df


def clean_mv(df):
    """ For loading and cleaning the multivariate summary. """
    # Change class counts to be a list rather than loads of columns
    cols = ['ClassCounts'] + [x for x in df.columns if 'Unnamed:' in x]
    df['ClassCounts'] = df[cols].apply(lambda x: [int(x) for x in list(x) if ~np.isnan(x)], axis=1)
    df.drop(cols[1:], axis=1, inplace=True)
    return df


def mv_benchmarks():
    """Benchmarks for Multivariate.

    Taken from: https://arxiv.org/pdf/1811.00075.pdf
    """
    benchmarks = {
        'ArticularyWordRecognition': 0.987,
        'AtrialFibrillation': 0.267,
        'BasicMotions': 1,
        'CharacterTrajectories': 0.989,
        'Cricket': 1,
        'DuckDuckGeese': 0.6,
        'EigenWorms': 0.618,
        'Epilepsy': 0.978,
        'ERing': 0.133,
        'EthanolConcentration': 0.323,
        'FaceDetection': 0.529,
        'FingerMovements': 0.55,
        'HandMovementDirection': 0.306,
        'Handwriting': 0.607,
        'Heartbeat': 0.717,
        'InsectWingbeat': 0.128,
        'JapaneseVowels': 0.959,
        'Libras': 0.894,
        'LSST': 0.575,
        'MotorImagery': 0.51,
        'NATOPS': 0.883,
        'PEMS-SF': 0.734,
        'PenDigits': 0.977,
        'PhonemeSpectra': 0.151,
        'RacketSports': 0.868,
        'SelfRegulationSCP1': 0.775,
        'SelfRegulationSCP2': 0.539,
        'SpokenArabicDigits': 0.967,
        'StandWalkJump': 0.333,
        'UWaveGestureLibrary': 0.903,
    }
    df = pd.DataFrame.from_dict(benchmarks, orient='index')
    df.columns = ['ACC']
    return df



if __name__ == '__main__':
    # Load full summaries
    # uv = pd.read_csv(DATA_DIR + '/raw/summaries/univariate.csv', encoding='latin-1', index_col=2)
    mv = pd.read_csv(DATA_DIR + '/raw/summaries/multivariate.csv', index_col=0)

    # Clean
    mv = clean_mv(mv)
    # uv = clean_uv(uv)

    # Get benchmarks for mv
    mv_benchmarks = mv_benchmarks()

    # Add type col to mv
    mv['Type'] = np.nan

    # Save
    # save_pickle(uv, DATA_DIR + '/interim/summaries/univariate.pkl')
    save_pickle(mv, DATA_DIR + '/interim/summaries/multivariate.pkl')

    # Scores -> DataFrame
    tt_split = pd.read_csv(DATA_DIR + '/raw/summaries/accuracies/singleTrainTest.csv', index_col=0)
    resample_split = pd.read_csv(DATA_DIR + '/raw/summaries/accuracies/resamples.csv', index_col=0)
    save_pickle(tt_split, DATA_DIR + '/interim/summaries/accuracies/train_test_split.pkl')
    save_pickle(resample_split, DATA_DIR + '/interim/summaries/accuracies/resample_split.pkl')

    save_pickle(mv_benchmarks, DATA_DIR + '/interim/summaries/accuracies/multivariate.pkl')
