MNIST Dataset Loader, Visualization, and CNN Training Script
=============================================================

This project demonstrates how to load MNIST handwritten digit data from IDX files,
apply optional data augmentation, visualize samples, and train a simple convolutional
neural network (CNN) classifier using PyTorch.

Features:
---------
- Loads training and testing datasets directly from IDX files using `idx2numpy`
- Implements a custom PyTorch `Dataset` class (`MnistDataset`)
- Applies optional data augmentation (random flips, inversion) to training data
- Visualizes a random sample grid of dataset images and saves it to disk
- Defines a small CNN model for digit classification
- Trains the model with SGD optimizer and learning rate scheduler
- Evaluates performance on a validation set each epoch
- Saves model checkpoints to the `models/` directory

Directory Structure:
--------------------
project/
├── dataset/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── models/          # Created automatically if not present
├── main.py
├── download.py
└── requirements.txt # torch not included

Dependencies:
-------------
- Python 3.10+
- torch
- torchvision
- numpy
- idx2numpy
- matplotlib
- tqdm

To install dependencies using the provided requirements.txt (excluding torch):
    pip install -r requirements.txt

Main Components:
----------------
1. show_images(images, output_path)
   - Displays and saves a grid of grayscale images.
   - images: list of torch.Tensor in shape (1, H, W)
   - output_path: path to save the image grid

2. MnistDataset(img_path, label_path, train=False)
   - Custom PyTorch Dataset for loading MNIST from IDX files.
   - train=True enables random flips and inversion augmentations
   - Returns: (image_tensor[float32, 1x28x28], label[int])

3. CNN
   - A simple convolutional neural network:
     - Conv2d(1 → 32), MaxPool(2)
     - Conv2d(32 → 64), MaxPool(2)
     - Fully connected layers: 64×7×7 → 128 → 10

4. eval(val_loader, model)
   - Evaluates the model on a validation DataLoader.
   - Returns:
     - Average loss (float)
     - Accuracy (float)

5. train(model, train_loader, val_loader, epochs)
   - Trains the CNN model.
   - Optimizer: SGD (default params)
   - Scheduler: StepLR with step size = epochs // 10
   - Prints training and validation metrics each epoch
   - Saves model checkpoints after each epoch

Execution Flow:
---------------
1. Run the dataset downloader:
       python download.py

2. Load training and validation datasets via MnistDataset

3. Visualize 100 random augmented images from the training set

4. Train the CNN for the specified number of epochs (default: 50)

5. Save model checkpoints after every epoch to models/

Dataset Source:
---------------
Available on Kaggle:
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Setup on WSL (Optional):
------------------------
- Install ROCm on WSL:
  https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html

- Install PyTorch on WSL:
  https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html

Author:
-------
Di Nguyen
