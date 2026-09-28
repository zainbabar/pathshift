# MedMNIST loading, splits, and data loaders.

from pathlib import Path

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader, TensorDataset


def get_device():
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

    data = np.load(root / 'pathmnist.npz')
    tensors = {name: torch.from_numpy(data[name]) for name in data.files}

    X_train = tensors['train_images'].to(torch.float32) / 255
    X_val = tensors['val_images'].to(torch.float32) / 255
    X_test = tensors['test_images'].to(torch.float32) / 255

    y_train = torch.flatten(tensors['train_labels']).long()
    y_val = torch.flatten(tensors['val_labels']).long()
    y_test = torch.flatten(tensors['test_labels']).long()

    X_train = torch.permute(X_train, (0, 3, 1, 2))
    X_val = torch.permute(X_val, (0, 3, 1, 2))
    X_test = torch.permute(X_test, (0, 3, 1, 2))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
