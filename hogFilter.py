import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math as m
from hog import hog
from spatialReliability import spatialMap
from scipy import signal


def hogFilter(hsv, bbox, cell_size=(8,8)):
    '''
    Calculates HOG features from the supplied HSV image

    Inputs:
    hsv - Video frame in hsv color space
    bbox - ROI to create filter from
    cell_size - Size of HOG cells

    Outputs:
    Each output array has 9 channels, for the different orientations of
    the gradients
    
    filt - Filters to be used in correlation
    feat - Features to apply next frames filter on
    exp - Expected response (filt convoluted with feat)
    learn - Channel Learning reliability (Highest activation in each channel of exp)
    '''
    x,y,w,h = bbox
    res = spatialMap(hsv, bbox)
    mask = np.zeros(hsv.shape,np.uint8)
    mask[:,:,0] = res
    mask[:,:,1] = res
    mask[:,:,2] = res

    maskedImg = cv2.bitwise_and(hsv, mask)

    patch = maskedImg[y:y+h,x:x+w]

# Filter should be the whole HOG image, but I get better results with a patch
    filt = hog(patch, cell_size)
    #filt = hog(maskedImg, cell_size)

    feat = hog(hsv, cell_size)

    maskedFilt = np.array(feat)

    feat_y, feat_x, _ = feat.shape
    
    res = cv2.resize(res,(feat_x, feat_y))
    #print(hsv.shape, feat.shape, res.shape)
    for i in range(0,9):
        maskedFilt[:,:,i] = np.multiply(feat[:,:,i],res)
    #filt = maskedFilt
    #print(filt.shape)
    
    #9 bins
    exp = np.zeros(feat.shape)
    learn = np.zeros(9)
    for i in range(0, 9):
        exp[:,:,i], learn[i] = convFilt(filt[:,:,i],feat[:,:,i])

    learn = cv2.normalize(learn, learn, 0, 1, cv2.NORM_MINMAX)
    
    return filt, feat, exp, learn

def convFilt(filt, feat):
    """
    Performs convolution in the fourier domain using scipy

    Inputs:
    filt - single channel, unnormalized
    feat - feature image
    
    Returns:
    exp - expected filter response
    peak - maximum response value before normalization
    """

    filt = filt[::-1,::-1]
    
    h = (filt-np.mean(filt))/max(np.sum(filt),1)

    exp = signal.fftconvolve(feat, h, mode='same')
    out = exp
    peak = np.max(exp)
    out = cv2.normalize(exp, out, 0, 1, cv2.NORM_MINMAX)
    
    return out, peak

