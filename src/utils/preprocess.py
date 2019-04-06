import sys
import os
import math
import scipy.ndimage
import numpy as np
import SimpleITK as sitk

def resample(image, spacing, new_spacing=[1.6, 0.7, 0.7], method = 'linear'):
    """
    Resample image given spacing (in mm) to new_spacing (in mm).
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

def getSpacing(image):
    return [image.GetSpacing()[2], image.GetSpacing()[0], image.GetSpacing()[1]]

def extend_image(image, val = 0, size = 512):
    """
    Resize <image> centered to <size> filling with extra values <val>
    """
    result = np.zeros(shape = (image.shape[0], size, size), dtype = np.int16)
    result.fill(val)

    x = int((size - image.shape[1])/2)
    y = int((size - image.shape[2])/2)
    result[:, x:(x+image.shape[1]), y:(y+image.shape[2])] = image
    return result

def crop_image(image, size = 512):
    cx = int(image.shape[1]/2)
    cy = int(image.shape[2]/2)
    new_img = image[:, cx - math.floor(size/2):cx + math.ceil(size/2), cy - math.floor(size/2):cy + math.ceil(size/2)]
    return new_img

def resize_image(image, size = 512):
    x = image.shape[1]
    if x < size:
        return extend_image(image, val = image[0, 0, 0], size = size)
    elif x > size:
        return crop_image(image, size = size)
    else:
        return image

def process_image(image_path, mask = False):
    image = sitk.ReadImage(image_path)
    imageNP = sitk.GetArrayViewFromImage(image)
    if mask:
        imageNP = (imageNP > 0).astype(np.int16)
    imageNP, _ = resample(image = imageNP, spacing = getSpacing(image), method = 'linear')
    return resize_image(imageNP)


if __name__== "__main__":
    subset = 'subset'
    # Paths for the server.
    image_path = os.path.join(os.path.expanduser('~'), 'LUNA')
    mask_path = os.path.join(os.path.expanduser('~'), 'LUNA', 'seg-lungs-LUNA16')
    image_processed_path = os.path.join(os.path.expanduser('~'), 'LUNA', 'preprocessed')
    mask_processed_path = os.path.join(os.path.expanduser('~'), 'LUNA', 'preprocessed', 'seg-lungs-LUNA16')
    # Paths for the PC
    #image_path = os.path.join(os.path.expanduser('~'), 'Documents', 'LUNA')
    #mask_path = os.path.join(os.path.expanduser('~'), 'Documents', 'LUNA', 'seg-lungs-LUNA16')
    #image_processed_path = os.path.join(os.path.expanduser('~'), 'Documents', 'LUNA', 'preprocessed')
    #mask_processed_path = os.path.join(os.path.expanduser('~'), 'Documents', 'LUNA', 'preprocessed', 'seg-lungs-LUNA16')
    
    report = open(os.path.join(os.path.expanduser('~'), 'preprocess.txt'), 'w+')
    for i in range(0, 10):
        print('Preprocessing subset', i, 'out of 9.')
        report.write('\n----------------------------------------------------------------------------------------------------------\n')
        report.write('----------------------------------------------------------------------------------------------------------\n')
        report.write('----------------------------------------------------------------------------------------------------------\n')
        report.write('----------------------------------------------------------------------------------------------------------\n')
        report.write('----------------------------------------------------------------------------------------------------------\n')
        report.write('preprocessing images in subset ' + str(i) + ' out of 9\n\n')
        path = os.path.join(image_path, subset + str(i))
        count = 0
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file_name in f:
                if '.mhd' in file_name:
                    count += 1
                    if count%10 == 0:
                        print('Image', count, 'out of 89.')
                    report.write('processing image: ' + str(file_name) + '\n')
                    report.flush()
                    processed_image = process_image(os.path.join(path, file_name))
                    processed_mask = process_image(os.path.join(mask_path, file_name), mask = True)
                    if processed_image.shape[1] != 512 or processed_image.shape[2] != 512:
                        print('ERROR, shape is', processed_image.shape, 'in image', file_name)
                        report.write('ERROR, shape is ' + processed_image.shape + ' in image ' + file_name + '\n')
                        report.flush()
                    if processed_mask.shape[1] != 512 or processed_mask.shape[2] != 512:
                        print('ERROR, shape is', processed_image.shape, 'in mask', file_name)
                        report.write('ERROR, shape is ' + processed_image.shape + ' in mask ' + file_name + '\n')
                        report.flush()
                    np.save(os.path.join(image_processed_path, subset + str(i), file_name.replace('.mhd', '')), processed_image)
                    np.save(os.path.join(mask_processed_path, file_name.replace('.mhd', '')), processed_mask)
        report.flush()