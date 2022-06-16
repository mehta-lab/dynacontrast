import os
import cv2
import natsort
import numpy as np


def within_range(r, pos):
    """ Check if a given position is in window

    Args:
        r (tuple): window, ((int, int), (int, int)) in the form of
            ((x_low, x_up), (y_low, y_up))
        pos (tuple): (int, int) in the form of (x, y)

    Returns:
        bool: True if `pos` is in `r`, False otherwise

    """
    if pos[0] >= r[0][1] or pos[0] < r[0][0]:
        return False
    if pos[1] >= r[1][1] or pos[1] < r[1][0]:
        return False
    return True


def check_segmentation_dim(segmentation):
    """ Check segmentation mask dimension.
    Add a background channel if n(channels)==1

    Args:
        segmentation: (np.array): segmentation mask for the frame

    """
    if segmentation.ndim == 5:
        return segmentation
    if segmentation.ndim == 4: # missing channel dim
        segmentation = segmentation[:, np.newaxis, ...]
    if segmentation.ndim == 3:
        segmentation = segmentation[np.newaxis, np.newaxis, ...]
    elif segmentation.ndim == 2:
        segmentation = segmentation[np.newaxis, np.newaxis, np.newaxis, ...]
    else:
        raise ValueError('segmentation mask must be 2-5D, not {}'.format(segmentation.ndim))
    return segmentation


def cv2_fn_wrapper(cv2_fn, mat, *args, **kwargs):
    """" A wrapper for cv2 functions

    Data in channel first format are adjusted to channel last format for
    cv2 functions
    """

    mat_shape = mat.shape
    x_size = mat_shape[-2]
    y_size = mat_shape[-1]
    _mat = mat.reshape((-1, x_size, y_size)).transpose((1, 2, 0))
    _output = cv2_fn(_mat, *args, **kwargs)
    _x_size = _output.shape[0]
    _y_size = _output.shape[1]
    output_shape = tuple(list(mat_shape[:-2]) + [_x_size, _y_size])
    output = _output.transpose((2, 0, 1)).reshape(output_shape)
    return output


def select_window(img, window, padding=0., skip_boundary=False):
    """ Extract image patch

    Patch of `window` will be extracted from `img`,
    negative boundaries are allowed (padded)

    Args:
        img (np.array): target image, size should be (c, z(1), x(2048), y(2048))
            TODO: size is hardcoded now
        window (tuple): area-of-interest for the patch, ((int, int), (int, int))
            in the form of ((x_low, x_up), (y_low, y_up))
        padding (float, optional): padding value for negative boundaries
        skip_boundary (bool, optional): if to skip patches whose edges exceed
            the image size (do not pad)

    Returns:
        np.array: patch-of-interest

    """
    if len(img.shape) == 4:
        n_channels, n_z, x_full_size, y_full_size = img.shape
    elif len(img.shape) == 3:
        n_channels, x_full_size, y_full_size = img.shape
        # add a z axis
        # img = np.expand_dims(img, 1)
    else:
        raise NotImplementedError("window must be extracted from raw data of 3 or 4 dims")
    # print(f"\nwindow selection, img.shape = {img.shape}\twindow.shape = {window}")

    if skip_boundary and ((window[0][0] < 0) or
                          (window[1][0] < 0) or
                          (window[0][1] > x_full_size) or
                          (window[1][1] > y_full_size)):
        return None

    if window[0][0] < 0:
        output_img = np.concatenate([padding * np.ones_like(img[..., window[0][0]:, :]),
                                     img[..., :window[0][1], :]], 1)
    elif window[0][1] > x_full_size:
        output_img = np.concatenate([img[..., window[0][0]:, :],
                                     padding * np.ones_like(img[..., :(window[0][1] - x_full_size), :])], 1)
    else:
        output_img = img[..., window[0][0]:window[0][1], :]

    if window[1][0] < 0:
        output_img = np.concatenate([padding * np.ones_like(output_img[..., window[1][0]:]),
                                     output_img[..., :window[1][1]]], 2)
    elif window[1][1] > y_full_size:
        output_img = np.concatenate([output_img[..., window[1][0]:],
                                     padding * np.ones_like(output_img[..., :(window[1][1] - y_full_size)])], 2)
    else:
        output_img = output_img[..., window[1][0]:window[1][1]]
    return output_img


def get_im_sites(input_dir):
    """
    Get sites (FOV names) from numpy files in the input directory
    Args:
        input_dir (str): input directory

    Returns:
        sites (list): sites (FOV names)

    """
    img_names = [file for file in os.listdir(input_dir) if file.endswith(".npy") & file.startswith('img_') & ('_NN' not in file)]
    sites = [os.path.splitext(img_name)[0] for img_name in img_names]
    sites = natsort.natsorted(list(set(sites)))
    return sites


def get_cell_rect_angle(tm):
    """ Calculate the rotation angle for long axis alignment

    Args:
        tm (np.array): target mask

    Returns:
        float: long axis angle

    """
    _, contours, _ = cv2.findContours(tm.astype('uint8'), 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    rect = cv2.minAreaRect(contours[np.argmax(areas)])
    w, h = rect[1]
    ang = rect[2]
    if w < h:
        ang = ang - 90
    return ang

