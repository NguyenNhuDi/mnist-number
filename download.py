import os
import kagglehub
from glob import glob

"""
Download MNIST dataset and move the files to "dataset"
"""

if __name__ == '__main__':

    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    dest_path = f'{os.path.abspath(os.curdir)}/dataset'

    os.system(f'rm -rf {dest_path}')

    os.system(f'mv {path} {dest_path}')

    for item in glob(f'{dest_path}/*'):
        if os.path.isdir(item):
            os.system(f'rm -rf {item}')

    print(f'Dataset is now in {dest_path}')
