import SimpleITK as sitk
import numpy as np 
import pandas as pd 
import numpy.fft as fp
from skimage.segmentation import slic, find_boundaries
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


def convert2rgb(img_arr):
    img_clip = np.clip(img_arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_clip)
    pil_img = pil_img.convert('RGB')
    img_rgb = np.asarray(pil_img).copy()
    return img_rgb


def overlay_images(background, boundaries):
    channels = list()
    cond = np.where(boundaries == 255)

    for ch in range(background.shape[2]):
        img_ch = background[...,ch]
        if ch == 0:
            img_ch[cond] = 255
        else:
            img_ch[cond] = 0
        img_ch = np.expand_dims(img_ch, axis=2)
        channels.append(img_ch)
    
    img_rgb = np.concatenate(tuple(channels), axis=2)
    img_pil = Image.fromarray(img_rgb)
    return img_pil


def segment_image(img_arr, compactness, segments, sigma=3):
    #Smooth image and resample
    img_smoothed, _ = smooth_image(img_arr, sigma=1)
    
    #Get segmentation with Slic
    segments_slic = slic(img_smoothed, n_segments=segments, compactness=compactness, sigma=3)

    #Get boundaries from segmentation
    boundaries = find_boundaries(segments_slic)
    boundaries = 255*boundaries

    #Convert to rgb for visualization
    img_rgb = convert2rgb(img_arr)

    #Overlay boundaries and rgb_img
    img_pil = overlay_images(img_rgb, boundaries)
    return img_pil