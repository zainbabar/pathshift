# load pathmnist and make our train / val / test data loaders

from pathlib import Path

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader, TensorDataset


def get_device():  # use the best one we have: cuda, then mps, then cpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_dataloaders(batch_size=128, root=Path('data')):
    # download the dataset the first time
    if not (root / 'pathmnist.npz').exists():
        root.mkdir(exist_ok=True)
        PathMNIST(split='train', download=True, root=root, size=28)  # downloads all 3 splits in one file

    data = np.load(root / 'pathmnist.npz')  # load file using numpy
    tensors = {name: torch.from_numpy(data[name]) for name in data.files}

    # pixels to float32, divide by 255 so everything is in a 0-1 range
    X_train = tensors['train_images'].to(torch.float32) / 255
    X_val = tensors['val_images'].to(torch.float32) / 255
    X_test = tensors['test_images'].to(torch.float32) / 255

    # flatten the targets so it's 1 vector of class indices for our loss function
    y_train = torch.flatten(tensors['train_labels']).long()
    y_val = torch.flatten(tensors['val_labels']).long()
    y_test = torch.flatten(tensors['test_labels']).long()

    # (N, 28, 28, 3) -> (N, 3, 28, 28), one 28x28 grid per colour channel for the conv layers
    X_train = torch.permute(X_train, (0, 3, 1, 2))
    X_val = torch.permute(X_val, (0, 3, 1, 2))
    X_test = torch.permute(X_test, (0, 3, 1, 2))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # only shuffle train, val and test order doesn't matter
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
