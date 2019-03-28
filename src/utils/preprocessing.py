import numpy as np
import scipy.ndimage

def resample(image, spacing, new_spacing=[1, 1, 1], method='nearest'):
    """
    Resample image given spacing (in mm) to new_spacing (in mm).
    
    by @enricca
    """
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    if method == 'cubic':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=3)
    elif method == 'quadratic':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=2)
    elif method == 'linear':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)
    elif method == 'nearest':
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=0)
    else:
        raise NotImplementedError("Interpolation method not implemented.")

    return image, new_spacing
