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

import torch.nn as nn
from torch.nn.functional import relu
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import idx2numpy as i2n
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader


# -------------------------------------------------------------------
# File paths for MNIST IDX files (train and test sets)
# -------------------------------------------------------------------
CURR_DIR = os.path.abspath(os.curdir)

TEST_IMG_PATH = f'{CURR_DIR}/dataset/t10k-images.idx3-ubyte'
TEST_LABEL_PATH = f'{CURR_DIR}/dataset/t10k-labels.idx1-ubyte'
TRAIN_IMG_PATH = f'{CURR_DIR}/dataset/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = f'{CURR_DIR}/dataset/train-labels.idx1-ubyte'

BATCH_SIZE = 64
NUM_WORKERS = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CRITERION = nn.CrossEntropyLoss()


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
        labels = i2n.convert_from_file(label_path)  # Shape: (N,)

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
        image = torch.from_numpy(image).unsqueeze(0).float()

        # Apply data augmentation if in training mode
        if self.train:
            transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomInvert(p=0.5)
            ])
            image = transforms(image)

        return image, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,
                               padding=1)  # -> (32, 28, 28)
        # -> (32, 14, 14)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                               padding=1)  # -> (64, 14, 14)
        # -> (64, 7, 7)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Classification layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all but batch dim
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x  # raw logits


def eval(val_loader: DataLoader, model: CNN):
    total_correct = 0
    total_loss = 0
    total = 0
    for data in val_loader:
        image, label = data
        image, label = image.to(DEVICE), label.to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            loss = CRITERION(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

        loss = total_loss / total
        accuracy = total_correct / total
        return loss, accuracy


def train(model: CNN, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    optimizer = torch.optim.SGD(model.parameters())

    total = 0
    total_correct = 0
    total_loss = 0

    best_loss = 0xffffffffff
    best_accuracy = -1
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=(epochs // 10))

    model.train()

    for epoch in range(epochs):
        for data in tqdm(train_loader):
            image, label = data
            image, label = image.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(image)
            loss = CRITERION(outputs, label)
            loss.backward()
            optimizer.step()

            total += image.size(0)
            _, predictions = outputs.max(1)
            total_correct += (predictions == label).sum()

            total_loss += loss.item() * image.size(0)

        model.eval()
        eval_loss, eval_accuracy = eval(val_loader, model)
        model.train()

        train_loss = total_loss / total
        train_accuracy = total_correct / total

        print(
            f'Train Accuracy: {train_accuracy:6.8f} --- Train Loss: {train_loss:6.8f}\nEval Accuracy: {eval_accuracy:6.8f} --- Eval Loss: {eval_loss:6.8f}')

        if eval_accuracy >= best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            best_loss = eval_loss if eval_loss <= best_loss else best_loss

        print(
            f'Best Accuracy: {best_accuracy:6.8f} --- Best Loss: {best_loss:6.8f}\nCurrent Epoch: {epoch} --- Best Epoch: {best_epoch}')
        

        print('\n')

        scheduler.step()

        torch.save(model, os.path.join(
            f'{CURR_DIR}/models', f'mnist-classifier-{eval_accuracy}-{train_accuracy}-.pt'))


# -------------------------------------------------------------------
# Main script execution
# -------------------------------------------------------------------
if __name__ == '__main__':
    try:
        os.mkdir('models')
    except FileExistsError:
        ...
    
    # Create dataset instance (training mode enabled for augmentation)
    train_set = MnistDataset(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    val_set = MnistDataset(TEST_IMG_PATH, TEST_LABEL_PATH, train=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = CNN()
    model.to(device=DEVICE)

    # Fetch first 100 transformed images
    images = [train_set[random.randint(
        0, train_set.__len__() - 1)][0] for i in range(100)]

    # Save visualization to output.png
    show_images(images, f'{CURR_DIR}/random_100.png')

    print(f'--- begin trainning ---')

    train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=50)
