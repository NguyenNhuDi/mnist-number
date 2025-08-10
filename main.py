import os
import random
import numpy as np
import idx2numpy as i2n
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


CURR_DIR = os.path.abspath(os.curdir)

TEST_IMG_PATH = f'{CURR_DIR}/dataset/t10k-images.idx3-ubyte'
TEST_LABEL_PATH = f'{CURR_DIR}/dataset/t10k-labels.idx1-ubyte'
TRAIN_IMG_PATH = f'{CURR_DIR}/dataset/train-images.idx3-ubyte'
TRAIN_IMG_PATH = f'{CURR_DIR}/dataset/train-labels.idx1-ubyte'


def show_images(images,  output_path):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for image in images:
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis('off')
        index += 1

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


class MnistDataset(Dataset):

    def __init__(self, img_path: str, label_path: str, train: bool = False) -> None:
        self.train = train

        images = i2n.convert_from_file(img_path)
        labels = i2n.convert_from_file(label_path)

        assert len(images) == len(
            labels), f'Length of images must match length of labels'

        self.dataset = [(image, label) for image, label in zip(images, labels)]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple[np.ndarray, int]:
        image, label = self.dataset[index]
        label = int(label)
        return image, label


if __name__ == '__main__':

    test_set = MnistDataset(TEST_IMG_PATH, TEST_LABEL_PATH)
    images = [test_set.__getitem__(i)[0] for i in range(100)]

    show_images(images, f'{CURR_DIR}/output.png')
