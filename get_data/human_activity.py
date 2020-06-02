from definitions import *
import sys
import urllib.request
import zipfile
import torch
from sklearn.model_selection import train_test_split
sys.path.append('../../..')


def download():
    """ Downloads the data to /raw/human_activity. """
    base_loc = DATA_DIR + '/raw/human_activity'
    loc = base_loc + '/human_activity.zip'
    if os.path.exists(loc):
        print('Path already exists at {}. If you wish to re-download you must delete this folder.'.format(loc))
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip'
    urllib.request.urlretrieve(url, loc)

    with zipfile.ZipFile(loc, 'r') as zip_ref:
        zip_ref.extractall(base_loc)


def read_file_data(filename, type_):
    with open(filename) as acc_file:
        _acc_file_data = acc_file.read().strip().split('\n')
        acc_file_data = []
        for line in _acc_file_data:
            acc_file_data.append(tuple(type_(i) for i in line.strip().split(' ')))
    return acc_file_data


def main(threshold=100, normed_length=200):
    """Returns the Human Activity dataset, after you've downloaded it. (Run `python download.py` first).

    Returns two tensors, X and y.
    X has shape (3795, 200, 6) corresponding to (batch, length, channel), and is comprised of normalised floats.
    y has shape (3795,) corresponding to (batch,), and is comprised of integers 0, 1, 2, 3, 4, 5.

    Note that the dataset needs shuffling before you do anything with it!
    """
    base_loc = DATA_DIR + '/raw/human_activity/RawData'
    labels_file_data = read_file_data(base_loc + '/labels.txt', int)

    X = []
    y = []

    last_experiment_number = None
    last_user_number = None
    for experiment_number, user_number, activity_number, start, end in labels_file_data:
        # There are 12 classes:
        # 1 Walking
        # 2 Walking upstairs
        # 3 Walking downstairs
        # 4 Sitting
        # 5 Standing
        # 6 Lieing down
        # 7 Standing to siting
        # 8 Sitting to standing
        # 9 Siting to lieing down
        # 10 Lieing down to sitting
        # 11 Standing to lieing down
        # 12 Lieing down to standing
        # But some have very few samples, and without them it's basically a balanced classification problem.
        if activity_number > 6:
            continue

        end += 1
        if experiment_number != last_experiment_number or user_number != last_user_number:
            acc_filename = 'acc_exp{:02}_user{:02}.txt'.format(experiment_number, user_number)
            gyro_filename = 'gyro_exp{:02}_user{:02}.txt'.format(experiment_number, user_number)
            acc_file_data = torch.tensor(read_file_data(base_loc + '/' + acc_filename, float))
            gyro_file_data = torch.tensor(read_file_data(base_loc + '/' + gyro_filename, float))
            # Is a tensor of shape (length, channels=6)
            both_data = torch.cat([acc_file_data, gyro_file_data], dim=1)
        last_experiment_number = experiment_number
        last_user_number = user_number

        # minimum length is 74
        # maximum length is 2032
        # I think what they did in the original dataset was split it up into pieces roughly 74 steps long. It's not
        # obvious that it's going to be that easy to learn from short series so here we split it up into pieces
        # 'normed_length' steps long, and apply fill-forward padding to the end if it's still at least of length
        # 'threshold'' and discard it if it's shorter. This doesn't affect much of our dataset.
        for start_ in range(start, end, normed_length):
            start_plus = start_ + normed_length
            if start_plus > end:
                too_short = True
                if start_plus - end < threshold:
                    continue  # skip data
                end_ = min(start_plus, end)
            else:
                too_short = False
                end_ = start_plus
            Xi = both_data[start_:end_]
            if too_short:
                Xi = torch.cat([Xi, Xi[-1].repeat(start_plus - end, 1)], dim=0)
            X.append(Xi)
            y.append(activity_number - 1)
    X = torch.stack(X, dim=0)
    y = torch.tensor(y)
    return X, y


if __name__ == '__main__':
    download()

    # Convert and save
    X, y = main(threshold=50, normed_length=74)
    idxs = train_test_split(list(range(X.shape[0])))
    save_pickle(X, DATA_DIR + '/processed/HumanActivity/data.pkl')
    save_pickle(y, DATA_DIR + '/processed/HumanActivity/labels.pkl')
    save_pickle(idxs, DATA_DIR + '/processed/HumanActivity/original_idxs.pkl')
