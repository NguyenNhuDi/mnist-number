"""
MNIST Dataset Loader and Visualization Script
----------------------------------------------
This script loads MNIST image and label data from IDX files,
applies optional random transformations for data augmentation,
and visualizes the results as a grid of images saved to disk.

Dependencies:
- torch
- torchvision
- numpy
- idx2numpy
- matplotlib
- Python 3.10+ (for type hints like list[torch.Tensor])

Author: Di Nguyen
"""

import os
import torch
import numpy as np
import idx2numpy as i2n
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2


# -------------------------------------------------------------------
# File paths for MNIST IDX files (train and test sets)
# -------------------------------------------------------------------
CURR_DIR = os.path.abspath(os.curdir)

TEST_IMG_PATH = f'{CURR_DIR}/dataset/t10k-images.idx3-ubyte'
TEST_LABEL_PATH = f'{CURR_DIR}/dataset/t10k-labels.idx1-ubyte'
TRAIN_IMG_PATH = f'{CURR_DIR}/dataset/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = f'{CURR_DIR}/dataset/train-labels.idx1-ubyte'  # FIXED variable name


# -------------------------------------------------------------------
# Function: show_images
# -------------------------------------------------------------------
def show_images(images: list[torch.Tensor], output_path: str) -> None:
    """
    Display a list of grayscale images in a grid and save to file.

    Args:
        images (list[torch.Tensor]):
            List of PyTorch tensors, each in shape (1, H, W)
            representing a single-channel grayscale image.
        output_path (str):
            Path where the output image grid will be saved.

    Notes:
        - Automatically handles torch.Tensors by converting to NumPy.
        - Squeezes the channel dimension for proper matplotlib display.
        - Saves the figure to disk instead of showing interactively.
    """
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1

    for image in images:
        plt.subplot(rows, cols, index)

        # Convert tensor to NumPy array if needed
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()

        # Remove channel dimension -> (H, W)
        image = image.squeeze(0)

        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis('off')
        index += 1

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------
# Class: MnistDataset
# -------------------------------------------------------------------
class MnistDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MNIST data from IDX files.

    Attributes:
        train (bool):
            Whether to apply random transformations for training data.
        dataset (list[tuple[np.ndarray, int]]):
            List of (image, label) pairs.
        length (int):
            Number of samples in the dataset.

    Methods:
        __len__():
            Returns the total number of samples.
        __getitem__(index):
            Returns the transformed image tensor and label.
    """

    def __init__(self, img_path: str, label_path: str, train: bool = False) -> None:
        """
        Initialize the MNIST dataset.

        Args:
            img_path (str):
                Path to the IDX file containing image data.
            label_path (str):
                Path to the IDX file containing label data.
            train (bool, optional):
                If True, apply data augmentation transforms. Defaults to False.
        """
        self.train = train

        # Load images and labels from IDX format
        images = i2n.convert_from_file(img_path)   # Shape: (N, H, W)
        labels = i2n.convert_from_file(label_path) # Shape: (N,)

        assert len(images) == len(labels), \
            f'Length of images must match length of labels'

        # Store as a list of (image, label) pairs
        self.dataset = [(image, label) for image, label in zip(images, labels)]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """
        Retrieve a sample from the dataset.

        Args:
            index (int):
                Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]:
                - Image as a PyTorch tensor of shape (1, H, W).
                - Corresponding label as an integer.
        """
        image, label = self.dataset[index]
        label = int(label)

        # Convert NumPy array to torch.Tensor with channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # Apply data augmentation if in training mode
        if self.train:
            transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomInvert(p=0.5)
            ])
            image = transforms(image)

        return image, label


# -------------------------------------------------------------------
# Main script execution
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Create dataset instance (training mode enabled for augmentation)
    test_set = MnistDataset(TEST_IMG_PATH, TEST_LABEL_PATH, train=True)

    # Fetch first 100 transformed images
    images = [test_set[i][0] for i in range(100)]

    # Save visualization to output.png
    show_images(images, f'{CURR_DIR}/output.png')