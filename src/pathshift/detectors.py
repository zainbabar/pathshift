import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pathshift.corruptions import add_noise, blur, reduce_contrast
from pathshift.data import get_dataloaders
from pathshift.model import CNNClassifier
from pathshift.train import device


# take our trained model and some data, and then get the embeddings
# we want to run it through our layers EXCEPT the final output layer
def get_embeddings(model, data_loader, corruption=None, severity=None):
    embedding_model = model.layers[:-1]  # every layer but the last, still a sequential model
    embedding_model.eval()
    embeddings = []
    with torch.no_grad():  # don't wanna track w/ computation graph
        for X_batch, _ in data_loader:
            if corruption is not None:  # corrupt if needed
                X_batch = corruption(X_batch, severity)
            embeddings.append(embedding_model(X_batch.to(device)))
    # merge per batch embeddings to 1 tensor for whole set, (N, 128)
    # cpu since the permutation test is a ton of tiny ops, faster on cpu than mps
    return torch.cat(embeddings).cpu()


def median_sigma(z):  # z is our pooled set of x and y, of shape (m+n, 128)
    distances = torch.cdist(z, z)  # pairwise distance between all points, (m+n, m+n) tensor
    mask = torch.eye(len(z), dtype=torch.bool)  # identity matrix of same size as distances, t/f
    dropped_diagonal = distances[~mask]  # drop the diagonal, distance from a point to itself is 0
    return dropped_diagonal.median()  # sigma = median distance, closer than this means closer than typical


def kernel_matrix(z, sigma):
    distances = torch.cdist(z, z)
    # gaussian kernel, similarity between every pair of points. 1 if the same, goes to 0 the further apart
    return torch.exp(-distances ** 2 / (2 * sigma ** 2))


def mmd2(K, m):  # takes similarity matrix between everything and size of X (m)
    n = len(K) - m  # the size of Y
    # when we stack the points together to pool, our matrix has comparisons between xx, yy, and xy
    # want to extract each of these comparison types for our mmd formula
    # XX is the first m rows and cols of K, YY is the rest, XY is x rows vs y cols
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]  # K[m:, :m] is the diff shape but has the same mean cuz same numbers

    # the diagonal for XX and YY compare the same points against themselves, so they give no useful info
    # they just inflate the mean, making things seem more similar than they really are
    # so we want to drop the diagonal from our mean calc
    XX_avg = (K_XX.sum() - K_XX.diagonal().sum()) / (m * (m - 1))
    YY_avg = (K_YY.sum() - K_YY.diagonal().sum()) / (n * (n - 1))

    # XY is fine, all points are different
    XY_avg = K_XY.mean()

    return XX_avg + YY_avg - 2 * XY_avg


def permutation_test(x, y, n_perms=500):
    z = torch.cat((x, y))  # pool points
    m = len(x)
    sigma = median_sigma(z)
    K = kernel_matrix(z, sigma)
    observed = mmd2(K, m)  # get our experimental mmd^2 that we want to test
    count = 0
    for _ in range(n_perms):
        shuffled_indexes = torch.randperm(len(z))  # shuffled indexes for rows and columns, diff every time
        # reorder rows, then reorder columns
        if mmd2(K[shuffled_indexes][:, shuffled_indexes], m) >= observed:
            count += 1  # luck mmd was >= our observed mmd

    return (count + 1) / (n_perms + 1)  # p-value, proportion of shuffled mmd that are >= original mmd


def sample(emb, k):
    return emb[torch.randperm(len(emb))[:k]]  # k random rows, no repeats


# one trial = one fake deployment batch, run a bunch and see how often the alarm goes off
def detection_rate(ref_emb, pool_emb, sample_size, n_trials, n_perms, alpha):
    alarms = 0
    for _ in range(n_trials):
        x = sample(ref_emb, sample_size)  # reference batch
        y = sample(pool_emb, sample_size)  # deployment batch
        if permutation_test(x, y, n_perms) < alpha:
            alarms += 1
    return alarms / n_trials  # fraction of trials that raised an alarm


# detection rate vs severity, one line per corruption, one plot per batch size
def plot_detection(detection, path):
    colors = {'noise': '#2a78d6', 'contrast': '#eb6834', 'blur': '#1baf7a'}  # same colors as the robustness plot
    markers = {'noise': 'o', 'contrast': 's', 'blur': '^'}
    sample_sizes = list(detection)
    fig, axes = plt.subplots(
        1, len(sample_sizes), figsize=(5 * len(sample_sizes), 4.5), sharey=True, squeeze=False
    )
    for ax, sample_size in zip(axes[0], sample_sizes):
        rates = detection[sample_size]
        for name in colors:
            ax.plot(
                range(1, 6),
                [rates[f'{name} {severity}'] for severity in range(1, 6)],
                color=colors[name],
                marker=markers[name],
                label=name,
            )
        # clean test and clean val don't have a severity, so they're horizontal lines
        ax.axhline(rates['clean test'], linestyle='--', color='gray', label='clean test (other center)')
        ax.axhline(rates['clean val'], linestyle=':', color='gray', label='clean val (false alarms)')
        ax.set_title(f'batches of {sample_size} images')
        ax.set_xticks(range(1, 6))
        ax.set_xlabel('severity')
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel('detection rate (p < 0.05)')

    # one legend under both plots so it doesn't cover any lines
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.suptitle('PathMNIST CNN: MMD shift detection')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)


if __name__ == '__main__':
    with open('checkpoints/config.json') as f:
        config = json.load(f)

    # load the trained model from train.py
    model = CNNClassifier(config['n_classes']).to(device)
    model.load_state_dict(torch.load('checkpoints/cnn.pt', map_location=device))
    train_loader, val_loader, test_loader = get_dataloaders(config['batch_size'])

    mmd_config = {'seed': 42, 'sample_sizes': [50, 200], 'n_perms': 500, 'n_trials': 100, 'alpha': 0.05}
    torch.manual_seed(mmd_config['seed'])

    val_emb = get_embeddings(model, val_loader)  # (10004, 128)
    test_emb = get_embeddings(model, test_loader)  # (7180, 128)

    # split val in half at random
    # reference half: every batch gets compared against this
    # held out half: where the clean val and corrupted batches come from, no shared images w/ the reference
    half = len(val_emb) // 2
    val_perm = torch.randperm(len(val_emb))
    ref_idx, held_idx = val_perm[:half], val_perm[half:]
    ref_emb = val_emb[ref_idx]

    # the pools we draw deployment batches from
    # we corrupt val and NOT test, test is already shifted so it'd get flagged even w/ no corruption
    pools = {
        'clean val': val_emb[held_idx],  # no shift, false alarm check
        'clean test': test_emb,  # real shift, different clinical center
    }
    corruptions = {'noise': add_noise, 'contrast': reduce_contrast, 'blur': blur}
    for name, corruption in corruptions.items():
        for severity in range(1, 6):
            torch.manual_seed(mmd_config['seed'])  # same noise every run
            corrupted_emb = get_embeddings(model, val_loader, corruption, severity)
            pools[f'{name} {severity}'] = corrupted_emb[held_idx]  # only keep the held out half

    # 17 pools x 100 trials x 500 shuffles per batch size, takes a few minutes
    detection = {}
    for sample_size in mmd_config['sample_sizes']:
        torch.manual_seed(mmd_config['seed'])
        detection[sample_size] = {}
        print(f'batches of {sample_size}')
        for name, pool_emb in pools.items():
            detection[sample_size][name] = detection_rate(
                ref_emb,
                pool_emb,
                sample_size,
                mmd_config['n_trials'],
                mmd_config['n_perms'],
                mmd_config['alpha'],
            )
            print(f'  {name:12} {detection[sample_size][name]:.2f}')

    # save the numbers
    Path('results').mkdir(exist_ok=True)
    with open('results/detection.json', 'w') as f:
        json.dump({'config': config, 'mmd_config': mmd_config, 'detection': detection}, f, indent=2)

    plot_detection(detection, 'results/detection.png')
