from definitions import *
import os
import pathlib
import sklearn.model_selection
import tarfile
import torch
import torchaudio
import urllib.request
from get_data.helpers import mkdir_if_not_exists

here = pathlib.Path(__file__).resolve().parent


def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir + '/' +  tensor_name) + '.pt')


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir + '/' + filename))
            tensors[tensor_name] = tensor_value
    return tensors


def download():
    # Make the speech_commands directory
    directory = DATA_DIR + '/raw/speech_commands'
    mkdir_if_not_exists(directory)

    loc = directory + '/speech_commands_data.tar.gz'
    extract_loc = DATA_DIR + '/raw/speech_commands/raw_speech_commands_data'
    if os.path.exists(extract_loc):
        print('Path already exists at {}. If you wish to re-download you must delete this folder.'.format(extract_loc))
        return
    if not os.path.exists(loc):
        urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
    with tarfile.open(loc, 'r') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, extract_loc)


def process_data():
    base_loc = DATA_DIR + '/raw/speech_commands/raw_speech_commands_data'
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = base_loc +'/'+ foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load_wav(loc + '/' + filename, channels_first=False,
                                           normalization=False)  # for forward compatbility if they fix it
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    # X is of shape (batch=34975, length=16000, channels=1)
    X = torchaudio.transforms.MFCC(log_mels=True,
                                   melkwargs=dict(n_fft=100, n_mels=32), n_mfcc=10)(X.squeeze(-1)).transpose(1, 2).detach()
    # X is of shape (batch=34975, length=321, channels=10). For some crazy reason it requires a gradient, so detach.

    train_X, val_X, test_X = split_data(X, y)
    train_y, val_y, test_y = split_data(y, y)

    return train_X, val_X, test_X, train_y, val_y, test_y


def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 512
    if 'pin_memory' not in kwargs:
        kwargs['pin_memory'] = True
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 6
    kwargs['batch_size'] = min(len(dataset), kwargs['batch_size'])
    return torch.utils.data.DataLoader(dataset, **kwargs)


def get_data():
    loc = DATA_DIR + '/raw/speech_commands/processed_speech_commands_data'
    try:
        tensors = load_data(loc)
        train_X = tensors['train_X']
        val_X = tensors['val_X']
        test_X = tensors['test_X']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
    except (KeyError, FileNotFoundError):
        download()
        train_X, val_X, test_X, train_y, val_y, test_y = process_data()
        if not os.path.exists(loc):
            os.mkdir(loc)
        save_data(loc, train_X=train_X, val_X=val_X, test_X=test_X, train_y=train_y, val_y=val_y, test_y=test_y)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    train_dataloader = dataloader(train_dataset)
    val_dataloader = dataloader(val_dataset)
    test_dataloader = dataloader(test_dataset)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    download()
    train_dl, val_dl, test_dl = get_data()
    train_dls = [train_dl, val_dl]

    train_data, test_data = torch.cat([x.dataset.tensors[0] for x in train_dls]), test_dl.dataset.tensors[0]
    train_labels, test_labels = torch.cat([x.dataset.tensors[1] for x in train_dls]), test_dl.dataset.tensors[1]
    num_train = train_data.size(0)
    num_test = test_data.size(0)
    original_idxs = [list(range(0, num_train)), list(range(num_train, num_train + num_test))]

    data = torch.cat([train_data, test_data])
    labels = torch.cat([train_labels, test_labels])

    print(num_test, num_train)

    save_pickle(data, DATA_DIR + '/processed/SpeechCommands/data.pkl')
    save_pickle(labels, DATA_DIR + '/processed/SpeechCommands/labels.pkl')
    save_pickle(original_idxs, DATA_DIR + '/processed/SpeechCommands/original_idxs.pkl')

