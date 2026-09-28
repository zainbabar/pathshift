import json
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics

from pathshift.data import get_dataloaders, get_device
from pathshift.model import CNNClassifier

device = get_device()


def train_one_epoch(model, optimizer, criterion, train_loader, epoch=1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Loss for epoch {epoch}: {total_loss / len(train_loader)}')


def train(model, optimizer, criterion, train_loader, n_epochs):
    for epoch in range(n_epochs):
        train_one_epoch(model, optimizer, criterion, train_loader, epoch + 1)


def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)

    return aggregate_fn(torch.stack(metrics))


if __name__ == '__main__':
    config = {'seed': 42, 'batch_size': 128, 'learning_rate': 1e-3, 'n_epochs': 20, 'n_classes': 9}
    torch.manual_seed(config['seed'])

    train_loader, val_loader, test_loader = get_dataloaders(config['batch_size'])

    model = CNNClassifier(config['n_classes']).to(device)
    xentropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train(model, optimizer, xentropy, train_loader, config['n_epochs'])

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=config['n_classes']).to(device)
    val_acc = evaluate(model, val_loader, accuracy).item()
    print(f'Val accuracy: {val_acc}')

    # save the trained weights, and the settings that made them
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cnn.pt')
    config['val_accuracy'] = val_acc
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=2)
