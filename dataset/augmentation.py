import cv2
import numpy as np
from scipy import ndimage
# import dask.delayed as delay
from utils.patch_utils import cv2_fn_wrapper

# @delay
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

# @delay
def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=1, col_axis=2, channel_axis=0,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 3D numpy array - a 2D image with one or more channels.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows (aka Y axis) in the input image.
                  Direction: left to right.
        col_axis: Index of axis for columns (aka X axis) in the input image.
                  Direction: top to bottom.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError("'row_axis', 'col_axis', and 'channel_axis'"
                         " must be distinct")

    # TODO: shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(
            f"Invalid axis' indices: {actual_indices - valid_indices}")

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError("Channels are allowed and the first and last dimensions.")

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_crop(img, crop_ratio = (0.6, 1)):
    # Note: image_data_format is 'channel_first'
    height, width = img.shape[1], img.shape[2]
    dy = int(np.random.uniform(crop_ratio[0], crop_ratio[1]) * height)
    dx = int(np.random.uniform(crop_ratio[0], crop_ratio[1]) * height)
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    img = img[:, y:(y+dy), x:(x+dx)]
    return cv2_fn_wrapper(cv2.resize, img, (height, width))


def random_intensity_jitter(img, mean_jitter, std_jitter):
    if mean_jitter == 0 and std_jitter == 0:
        return img
    img_j = []
    for im in img:
        mean_offset = np.random.uniform(-mean_jitter, mean_jitter)
        std_scale = 1 + np.random.uniform(-std_jitter, std_jitter)
        im = unzscore(im, mean_offset, std_scale)
        img_j.append(im)
    return np.stack(img_j)


def augment_img(img, rotate_range=180, zoom_range=(1, 1), crop_ratio = (0.6, 1), intensity_jitter=(0.5, 0.5)):
# def augment_img(img, rotate_range=180, zoom_range=(1, 1), crop_ratio=(0.6, 1), intensity_jitter=(0, 0)):
# def augment_img(img, rotate_range=180, zoom_range=(1, 1), crop_ratio=(1, 1), intensity_jitter=(0.5, 0.5)):
    """Data augmentation with flipping and rotation"""
    # TODO: Rewrite with torchvision transform
    # img = np.asarray(img)
    img = random_intensity_jitter(img, intensity_jitter[0], intensity_jitter[1])
    if crop_ratio[0] != 1 or crop_ratio[1] != 1:
        img = random_crop(img, crop_ratio=crop_ratio)
    flip_idx = np.random.choice([0, 1, 2])
    if flip_idx != 0:
        img = np.flip(img, axis=flip_idx)
    theta = np.random.uniform(-rotate_range, rotate_range)
    zoom = np.random.uniform(zoom_range[0], zoom_range[1])
    img = apply_affine_transform(img, zx=zoom, theta=theta,
                                zy=zoom, fill_mode='constant', cval=0., order=1)
    # rot_idx = int(np.random.choice([0, 1, 2, 3]))
    # img = np.rot90(img, k=rot_idx, axes=(1, 2))
    return img

def unzscore(im_norm, mean, std):
    """
    Revert z-score normalization applied during preprocessing. Necessary
    before computing SSIM

    :param input_image: input image for un-zscore
    :return: image at its original scale
    """

    im = im_norm * (std + np.finfo(float).eps) + mean

    return im