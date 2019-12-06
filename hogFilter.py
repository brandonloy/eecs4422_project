import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math as m
from hog import hog
from spatialReliability import spatialMap
from scipy import signal


def hogFilter(hsv, bbox, cell_size=(8,8)):
    x,y,w,h = bbox
    res = spatialMap(hsv, bbox)
    mask = np.zeros(hsv.shape,np.uint8)
    mask[:,:,0] = res
    mask[:,:,1] = res
    mask[:,:,2] = res
##    print(res.dtype)
##    print(hsv.dtype)
##    print(mask.dtype)
    maskedImg = cv2.bitwise_and(hsv, mask)
##    plt.imshow(mask)
##    plt.show()
    patch = maskedImg[y:y+h,x:x+w]

    #cv2.imshow('',res)
    #plt.imshow(res)
    #plt.show()

# Filter should be the whole HOG image, but i started off with a patch
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

