import torch
import torchvision.transforms.v2 as T


def sev_dist(severity):
    vals = [0.02, 0.03, 0.04, 0.05, 0.06]
    return vals[severity - 1]


def add_noise(images, severity):
    return (images + (sev_dist(severity) * torch.randn_like(images))).clamp(0, 1)


def reduce_contrast(images, severity):
    mean = images.mean(dim=(2, 3), keepdim=True)  # mean per channel per image, shape (n, 3, 1, 1)
    images = mean + (1 - 0.15 * severity) * (images - mean)
    return images


def blur(images, severity):
    sigmas = [0.4, 0.5, 0.6, 0.8, 1.0]
    blur = T.GaussianBlur(
        kernel_size=(2 * severity + 1, 2 * severity + 1), sigma=sigmas[severity - 1]
    )
    return blur(images)
