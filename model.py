import SimpleITK as sitk
import numpy as np 
import pandas as pd 
import numpy.fft as fp
from skimage.segmentation import slic, mark_boundaries
from PIL import Image
from skimage.transform import rescale
from scipy import ndimage


def read_img(file_path):
    img = sitk.ReadImage(file_path)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr.reshape((img_arr.shape[1], img_arr.shape[2]))
    return img_arr
        
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def smooth_image(img_arr, sigma=3):
    freq = fp.fft2(img_arr)
    freq_gaussian = ndimage.fourier_gaussian(freq, sigma=sigma)
    im = fp.ifft2(freq_gaussian)
    return im.real, im.imag

def resample_image(img_arr, scale):
    img_resized = rescale(img_arr, scale= scale, anti_aliasing=True)
    return img_resized

def segment_image(img_arr, compactness, segments, scale, sigma=3):
    img_smoothed, _ = smooth_image(img_arr, sigma=sigma)
    img_resized = resample_image(img_smoothed, scale=scale)
    mean = img_resized.mean()
    std = img_resized.std()
    img_norm = (img_resized - mean) / std
    img_norm = np.expand_dims(img_norm, axis=2)
    img_norm = np.concatenate((img_norm, img_norm, img_norm), axis=2)
    segments_slic = slic(img_norm, n_segments=segments, compactness=compactness, sigma=1)
    boundaries = mark_boundaries(img_norm, segments_slic, color=(1,0,0))
    segmentation = np.clip((boundaries*std) + mean, 0 , 255).astype(np.uint8)
    im_pil = Image.fromarray(segmentation, 'RGB')
    return im_pil