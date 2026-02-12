import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.transform import resize
from skimage.filters import sobel
from skimage.color import rgb2lab, gray2rgb
from skimage.util import img_as_float

import matplotlib.pyplot as plt
from typing import Callable


def flatten(img: np.ndarray):
    """
    Wrapper around array flatten so I can
    overload it

    :param img: Description
    :type img: np.ndarray
    """
    return img.flatten()


def edge_magnitude(img):
    if img.ndim == 3:
        return flatten(np.sum(np.abs(sobel(img)), axis=2))
    return flatten(np.abs(sobel(img)))


def luminance(img: np.ndarray):
    if img.ndim == 2:
        return img
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def pixel_sorting(
    img: np.ndarray,
    target: np.ndarray,
    score_func: Callable = flatten,
    colour_func: Callable = luminance,
) -> tuple[np.ndarray, tuple[np.ndarray]]:
    """
    Docstring for pixel_sorting
    This function calculated the way to make 1 image look like another using
    pallette sorting

    :param img: Description
    :type img: np.ndarray
    :param target: Description
    :type target: np.ndarray
    """

    if target.ndim == 3 and img.ndim == 2:
        img = gray2rgb(img)

    input_img = np.array(resize(img, target.shape, anti_aliasing=True))

    # Handle Colour
    if input_img.ndim > 2:
        palette = colour_func(input_img)
    else:
        palette = input_img

    if target.ndim > 2:
        template = colour_func(target)
    else:
        template = target

    input_vec = palette.flatten()

    # Compute features to sort on
    input_scores = score_func(palette)
    target_scores = score_func(template)

    input_idx = np.argsort(input_scores)
    target_idx = np.argsort(target_scores)

    # reshape result
    if input_img.ndim <= 2:
        result = np.zeros_like(input_vec)
        result[target_idx] = input_vec[input_idx]
        return result.reshape(input_img.shape), input_idx, target_idx
    input_y, input_x = np.unravel_index(input_idx, palette.shape)
    target_y, target_x = np.unravel_index(target_idx, template.shape)
    result = np.zeros_like(target)
    result[target_y, target_x, :] = input_img[input_y, input_x, :]
    return (
        result,
        (input_y, input_x, target_y, target_x),
    )


if __name__ == "__main__":
    from skimage import data
    from skimage.io import imread
    from skimage.color import rgb2gray

    img = img_as_float(imread("assets/mylove.jpg"))
    pallette = img_as_float(data.cat())
    result, _ = pixel_sorting(pallette, img)

    plt.subplot(121)
    plt.imshow(result)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()
