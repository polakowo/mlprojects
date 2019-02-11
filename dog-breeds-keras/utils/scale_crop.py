import numpy as np
from PIL import Image


def scale_img(img, img_shape):
    """Downscale/upscale the image"""

    # resize the longest side to the input size
    img.thumbnail(img_shape, Image.ANTIALIAS)

    # resize the shortest side to the input size
    if img.size[0] < img_shape[0]:
        wpct = img_shape[0] / img.size[0]
        hsize = int(img.size[1] * wpct)
        img = img.resize((img_shape[0], hsize), Image.ANTIALIAS)

    if img.size[1] < img_shape[1]:
        h_pct = img_shape[1] / img.size[1]
        wsize = int(img.size[0] * h_pct)
        img = img.resize((wsize, img_shape[1]), Image.ANTIALIAS)

    return img


def center_crop_img(img, img_shape):
    """Crop the image at center"""

    left = np.floor((img.size[0] - img_shape[0]) / 2)
    top = np.floor((img.size[1] - img_shape[1]) / 2)
    right = np.floor((img.size[0] + img_shape[0]) / 2)
    bottom = np.floor((img.size[1] + img_shape[1]) / 2)

    img = img.crop((left, top, right, bottom))
    return img
