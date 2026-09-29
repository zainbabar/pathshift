# corruptions that could come up w/ a scanner: sensor noise, staining / exposure differences, focus problems
# every function takes a batch of images (n, 3, 28, 28) in [0, 1] and a severity from 1 to 5

import torch
import torchvision.transforms.v2 as T


def sev_dist(severity):
    vals = [0.02, 0.03, 0.04, 0.05, 0.06]  # mapping of values for adjustable severity (1-5)
    return vals[severity - 1]


def add_noise(images, severity):
    # randn_like gives random values (mean 0, var 1) w/ the same shape as the images, scaled by severity
    # clamp the output so the pixel intensities don't go out of our range
    return (images + (sev_dist(severity) * torch.randn_like(images))).clamp(0, 1)


def reduce_contrast(images, severity):
    # shift every pixel towards the mean of its image, so bright and dark pixels get closer together
    mean = images.mean(dim=(2, 3), keepdim=True)  # mean per channel per image, shape (n, 3, 1, 1)
    images = mean + (1 - 0.15 * severity) * (images - mean)  # keeps 85% of the contrast at sev 1, 25% at sev 5
    return images


def blur(images, severity):
    sigmas = [0.4, 0.5, 0.6, 0.8, 1.0]  # bigger sigma = blurrier
    # built in torchvision gaussian blur, kernel gets bigger w/ severity so it fits the sigma
    blur = T.GaussianBlur(
        kernel_size=(2 * severity + 1, 2 * severity + 1), sigma=sigmas[severity - 1]
    )
    return blur(images)
