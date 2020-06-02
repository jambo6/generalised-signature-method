import sys
sys.path.append('../../..')


from definitions import DATA_DIR

import os
import urllib.request
import tarfile



def main():
    base_loc = DATA_DIR + '/urban_sound'
    loc = base_loc + '/UrbanSound8K.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)

    url = 'https://goo.gl/8hY5ER'
    urllib.request.urlretrieve(url,loc)

    tar = tarfile.open(loc, "r:gz")
    tar.extractall(base_loc)
    tar.close()


if __name__ == '__main__':
    main()
