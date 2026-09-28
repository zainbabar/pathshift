import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchmetrics

from pathshift.corruptions import add_noise, blur, reduce_contrast
from pathshift.data import get_dataloaders
from pathshift.model import CNNClassifier
from pathshift.train import device, evaluate


def evaluate_corrupted(
    model, data_loader, metric_fn, corruption, severity, aggregate_fn=torch.mean
):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = corruption(X_batch, severity)  # corrupt the batch first
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)

    return aggregate_fn(torch.stack(metrics))


if __name__ == '__main__':
    with open('checkpoints/config.json') as f:
        config = json.load(f)

    # load the trained model from train.py
    model = CNNClassifier(config['n_classes']).to(device)
    model.load_state_dict(torch.load('checkpoints/cnn.pt', map_location=device))

    train_loader, val_loader, test_loader = get_dataloaders(config['batch_size'])
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=config['n_classes']).to(device)

    # clean accuracy is our baseline
    clean_val = evaluate(model, val_loader, accuracy).item()
    clean_test = evaluate(model, test_loader, accuracy).item()
    print(f'clean val: {clean_val}, clean test: {clean_test}')

    corruptions = {'noise': add_noise, 'contrast': reduce_contrast, 'blur': blur}

    results = {}
    for name, corruption in corruptions.items():
        results[name] = []
        for severity in range(1, 6):
            torch.manual_seed(config['seed'])  # same noise every run
            acc = evaluate_corrupted(model, test_loader, accuracy, corruption, severity)
            results[name].append(acc.item())
        print(name, results[name])

    # save the numbers
    Path('results').mkdir(exist_ok=True)
    with open('results/robustness.json', 'w') as f:
        json.dump(
            {
                'config': config,
                'clean_val': clean_val,
                'clean_test': clean_test,
                'corrupted_test': results,
            },
            f,
            indent=2,
        )

    # accuracy vs severity, one line per corruption, with clean accuracy as the baseline
    colors = {'noise': '#2a78d6', 'contrast': '#eb6834', 'blur': '#1baf7a'}
    markers = {'noise': 'o', 'contrast': 's', 'blur': '^'}
    for name, accs in results.items():
        plt.plot(range(1, 6), accs, color=colors[name], marker=markers[name], label=name)
    plt.axhline(clean_test, linestyle='--', color='gray', label='clean')
    plt.axhline(1 / 9, linestyle=':', color='gray', label='chance')
    plt.ylim(0, 1)
    plt.xticks(range(1, 6))
    plt.xlabel('severity')
    plt.ylabel('test accuracy')
    plt.title('PathMNIST CNN: test accuracy under corruption')
    plt.legend()
    plt.savefig('results/robustness.png', dpi=200)
