import sys

sys.path.append('../../..')

from definitions import DATA_DIR

import torch
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedKFold


def read_file_data(filename):

    raw_file_data = sf.read(filename)[0]
    # Convert data to be 2-dimensional. If only one channel, copy the channel in the other dimension.

    if len(raw_file_data.shape)==1:
        file_data=np.zeros((raw_file_data.shape[0],2))
        file_data[:,0]=raw_file_data
        file_data[:,1]=raw_file_data
    else:
        file_data=raw_file_data
    # Subample so that the series are of the order of 2000 points
    # subsampling=math.ceil(raw_file_data.shape[0]/2000)
    # file_data=file_data[::subsampling,:]
    return(file_data)


def main(normed_length=2000):
    labels_df=pd.read_csv(DATA_DIR + '/raw/urban_sound/UrbanSound8K/metadata/UrbanSound8K.csv')
    labels_df['filename'] = (labels_df['fold'].apply(lambda x: DATA_DIR+'/raw/urban_sound/UrbanSound8K/audio/fold'
        +str(x) +'/')+labels_df['slice_file_name'].apply(str))

    X = []
    y = []

    for i in range (labels_df.shape[0]):
    # for i in range(10):
        filename=labels_df['filename'].iloc[i]
        Xi=torch.tensor(read_file_data(filename))

        # If normalising length then do thisj
        if normed_length is not False:
            if Xi.shape[0]>normed_length:
                Xi=Xi[:normed_length,:]
            else:
                Xi=torch.cat([Xi, Xi[-1].repeat(normed_length - Xi.shape[0], 1)], dim=0)

        # Into list
        X.append(Xi)
        y.append(labels_df['classID'].iloc[i])
    if normed_length is not False:
        X = torch.stack(X, dim=0)
    y = torch.tensor(y)
    return X, y
    

if __name__ == '__main__':
    from definitions import *
    from sklearn.model_selection import train_test_split
    X, y = main(normed_length=False)

    idxs = train_test_split(list(range(X.shape[0])))

    save_pickle(X, DATA_DIR + '/interim/other/urban_sound/data.pkl')
    save_pickle(y, DATA_DIR + '/interim/other/urban_sound/labels.pkl')
