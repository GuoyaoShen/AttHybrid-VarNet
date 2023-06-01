import contextlib

import numpy as np
import torch
import operator
from scipy.stats import norm


@contextlib.contextmanager
def temp_seed(rng, seed):
    """
    fixed random function given seed

    :param rng: numpy random function
    :param seed: int, seed number
    :return: random function with given random seed
    """
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


def center_crop_np(img, bounding):
    """
    center crop an image given bounding size

    :param img: image of size (..., H, W)
    :param bounding: int, bounding size for center cropping
    :return: center crop image with size (..., bounding, bounding)
    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


class RandomMaskGaussian:
    def __init__(
            self,
            acceleration=4,
            center_fraction=0.08,
            size=(1, 256, 256),
            seed=None,
            mean=(0, 0),
            cov=[[1, 0], [0, 1]],
            concentration=3,
            patch_size=4,
    ):
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.size = size
        self.seed = seed
        self.mean = mean
        self.cov = cov
        self.concentration = concentration
        self.patch_size = patch_size

    def __call__(self):
        return random_mask_gaussian(
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            size=self.size,
            seed=self.seed,
            mean=self.mean,
            cov=self.cov,
            concentration=self.concentration,
            patch_size=self.patch_size,
        )


def random_mask_gaussian(
        acceleration=4,
        center_fraction=0.08,
        size=(16, 320, 320),
        seed=None,
        mean=(0, 0),
        cov=[[1, 0], [0, 1]],
        concentration=3,
        patch_size=4,
):
    """
    random_mask_gaussian creates a sub-sampling gaussian mask of a given shape.

    :param acceleration: float, undersample percentage 4X fold or 8X fold, default 4
    :param center_fraction: float, fraction of square center area left unmasked, defualt 0.08
    :param size: [B, H, W], output size for random gaussian mask, default [16, 320, 320]
    :param seed: None, int or [int, ...], seed for the random number generator. Setting the seed ensures the same mask
                is generated each time for the same seed number. The random state is reset afterwards. None for totally
                random, int for fixed seed across different batches, list of int for fixed seed of each slices in each
                batches. Default None
    :param mean: optional [int, int], gaussian mean on H, W channel. default [0, 0]
    :param cov: optional 2X2 gaussian covariance matrix on H, W channel. default [[1, 0], [0, 1]], note it assume
                independent dimensional covariance
    :param concentration: optional int, scale which indicates the size of area to concentrate on. default 3
    :param patch_size: optional int, size of each square pixel-wise mask, default 4
    :return mask, a np array of the specified shape. Its shape should be
            (batch_size, crop_size, crop_size) and the two channels are the same.


    """

    B, H, W = size
    if H != W:
        raise Exception("different height and width of the mask setting")

    if isinstance(seed, int):
        seed_list = seed * (np.arange(B) + 1)
    elif isinstance(seed, list):
        if len(seed) != B:
            raise Exception("different seed list length and batch size")
        else:
            seed_list = np.array(seed)
    else:
        seed_list = np.array([None] * B)

    rng = np.random
    crop_size = int(H / patch_size)
    margin = patch_size * 2
    half_size = crop_size / 2

    cdf_lower_limit = norm(mean[0], cov[0][0]).cdf(-concentration)
    cdf_upper_limit = norm(mean[1], cov[1][1]).cdf(concentration)
    probability = cdf_upper_limit - cdf_lower_limit

    num_pts = int(crop_size * crop_size / (acceleration * probability * probability))
    num_low_freqs = int(round(crop_size * center_fraction))
    pad = (crop_size - num_low_freqs + 1) // 2
    masks = np.zeros((B, H, W))

    for i in range(B):
        with temp_seed(rng, seed_list[i]):
            # gaussian distribution index

            gauss_np = rng.multivariate_normal(mean, cov, num_pts)
            gauss_np = gauss_np * (half_size / concentration) + half_size + margin / 2
            gauss_np = np.round(gauss_np).astype(int)
            gauss_np = np.clip(gauss_np, 0, crop_size + int(margin / 2))

            # apply gaussian index on mask
            mask = np.zeros((crop_size + margin, crop_size + margin))
            mask[gauss_np.transpose()[0], gauss_np.transpose()[1]] = 1.0
            mask = center_crop_np(mask, (crop_size, crop_size))

            # reset center square to unmasked
            mask[pad: pad + num_low_freqs, pad: pad + num_low_freqs] = 1.0
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, scale_factor=patch_size,
                                                   mode='nearest')
            mask = mask.squeeze(0).squeeze(0)
            masks[i] = mask.numpy()
    return masks
