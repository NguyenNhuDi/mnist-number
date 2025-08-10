"""
MNIST Dataset Loader, Visualization, and CNN Training Script
-------------------------------------------------------------

This script demonstrates:
    • Loading MNIST handwritten digit data from IDX files
    • Applying optional random transformations for data augmentation
    • Visualizing samples as an image grid
    • Defining and training a small CNN classifier using PyTorch
    • Evaluating the model on a validation set
    • Saving training checkpoints

Author:
    Di Nguyen
"""

import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import idx2numpy as i2n
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.nn.functional import relu
from torchvision.transforms import v2
from torch.utils.data import DataLoader


# -------------------------------------------------------------------
# File paths and configuration
# -------------------------------------------------------------------
CURR_DIR = os.path.abspath(os.curdir)

# Paths to IDX format files for training and test datasets
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
    Save a grid visualization of grayscale images to disk.

    Args:
        images (list[torch.Tensor]):
            List of images in shape (1, H, W), usually from a dataset.
        output_path (str):
            File path where the output grid image will be saved.

    Notes:
        - Automatically handles torch tensors (detaching and converting to NumPy).
        - Squeezes the channel dimension for proper matplotlib display.
        - Uses a fixed grid layout with 5 columns and enough rows for all images.
    """
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1

    for image in images:
        plt.subplot(rows, cols, index)

        # Convert to NumPy if it's a torch.Tensor
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()

        # Remove channel dimension
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
        train (bool): Whether data augmentation is applied.
        dataset (list[tuple[np.ndarray, int]]): List of (image, label) pairs.
        length (int): Total number of samples.

    Example:
        dataset = MnistDataset(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, train=True)
        image, label = dataset[0]
    """

    def __init__(self, img_path: str, label_path: str, train: bool = False) -> None:
        """
        Initialize the MNIST dataset loader.

        Args:
            img_path (str): Path to IDX file containing images.
            label_path (str): Path to IDX file containing labels.
            train (bool): If True, enable random augmentation.
        """
        self.train = train

        # Load IDX files
        images = i2n.convert_from_file(img_path)   # Shape: (N, H, W)
        labels = i2n.convert_from_file(label_path) # Shape: (N,)

        assert len(images) == len(labels), \
            f"Image count {len(images)} does not match label count {len(labels)}"

        self.dataset = [(image, label) for image, label in zip(images, labels)]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """
        Retrieve a single sample.

        Args:
            index (int): Dataset index.

        Returns:
            tuple:
                - image (torch.Tensor): Shape (1, H, W), dtype float32, values in [0, 255]
                - label (int): Digit label in range 0–9
        """
        image, label = self.dataset[index]
        label = int(label)

        # Add channel dimension and convert to float32
        image = torch.from_numpy(image).unsqueeze(0).float()

        # Apply augmentations if training
        if self.train:
            transforms = v2.Compose([
                v2.RandomRotation(degrees=10),
                v2.RandomAffine(degrees=0, translate=(0.5, 0.5)),
                v2.RandomInvert(p=0.5)
            ])
            image = transforms(image)

        return image, label


class BasicBlock(nn.Module):
    """
    Basic Residual Block with two conv layers and skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# -------------------------------------------------------------------
# Function: eval
# -------------------------------------------------------------------
def eval(val_loader: DataLoader, model: CNN):
    """
    Evaluate the model on a validation set.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (CNN): Trained model.

    Returns:
        tuple:
            - loss (float): Average cross-entropy loss.
            - accuracy (float): Fraction of correctly predicted samples.
    """
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

    return total_loss / total, total_correct / total


# -------------------------------------------------------------------
# Function: train
# -------------------------------------------------------------------
def train(model: CNN, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    """
    Train a CNN model on MNIST.

    Args:
        model (CNN): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of training epochs.

    Saves:
        Model checkpoints are saved each epoch in `models/` directory.
    """
    optimizer = torch.optim.SGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(epochs // 10))

    total = 0
    total_correct = 0
    total_loss = 0
    best_loss = float("inf")
    best_accuracy = -1
    best_epoch = 0

    model.train()

    for epoch in range(epochs):
        for image, label in tqdm(train_loader):
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

        # Validation phase
        model.eval()
        eval_loss, eval_accuracy = eval(val_loader, model)
        model.train()

        train_loss = total_loss / total
        train_accuracy = total_correct / total

        print(f"Train Accuracy: {train_accuracy:.6f} --- Train Loss: {train_loss:.6f}")
        print(f"Eval Accuracy:  {eval_accuracy:.6f} --- Eval Loss:  {eval_loss:.6f}")

        if eval_accuracy >= best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            best_loss = min(best_loss, eval_loss)

        print(f"Best Accuracy: {best_accuracy:.6f} --- Best Loss: {best_loss:.6f}")
        print(f"Current Epoch: {epoch} --- Best Epoch: {best_epoch}\n")

        scheduler.step()

        torch.save(model, os.path.join(
            f'{CURR_DIR}/models', f'mnist-classifier-{eval_accuracy}-{train_accuracy}.pt'
        ))


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == '__main__':
    try:
        os.mkdir('models')
    except FileExistsError:
        pass

    # Load datasets and data loaders
    train_set = MnistDataset(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_set = MnistDataset(TEST_IMG_PATH, TEST_LABEL_PATH, train=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize and move model to device
    model = CNN()
    model.to(device=DEVICE)

    # Visualize 100 random training samples with augmentation
    images = [train_set[random.randint(0, len(train_set) - 1)][0] for _ in range(100)]
    show_images(images, f'{CURR_DIR}/random_100.png')

    print("--- Begin Training ---")
    train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=200)
