import os
import urllib
import zipfile


def save_zip(url, loc):
    if not os.path.exists(loc):
        urllib.request.urlretrieve(url, loc)


def unzip(file, loc):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(loc)


def mkdir_if_not_exists(loc):
    if not os.path.exists(loc):
        os.mkdir(loc)
